module.exports = ( function ( mw, wb, $, OO ) {
	'use strict';

	var defaultConfig = {
		CACHED_STATUSES: 'violation|warning|suggestion|bad-parameters',
		WBCHECKCONSTRAINTS_MAX_ID_COUNT: 50
	};

	function SELF( config ) {
		this.config = $.extend( {}, defaultConfig, config );
	}

	SELF.prototype.setEntity = function ( entity ) {
		this._entity = entity;
	};

	SELF.prototype.getEntity = function () {
		return this._entity;
	};

	SELF.prototype.defaultBehavior = function () {
		var entityId = mw.config.get( 'wbEntityId' );

		if ( entityId === null || mw.config.get( 'wgMFMode' ) || !mw.config.get( 'wbIsEditView' ) ) {
			// no entity, mobile frontend, or not editing (diff, oldid, …) – skip
			return;
		}

		mw.loader.using( [
			'mediawiki.api',
			'oojs-ui-core',
			'oojs-ui-widgets',
			'oojs-ui.styles.icons-alerts',
			'wikibase.quality.constraints.icon',
			'wikibase.quality.constraints.ui',
			'wikibase.EntityInitializer'
		] ).done( function () {
			var api = new mw.Api(),
				lang = mw.config.get( 'wgUserLanguage' );

			wb.EntityInitializer.newFromEntityLoadedHook().getEntity().done( function ( entity ) {
				this.setEntity( entity );
				mw.hook( 'wikibase.entityPage.entityView.rendered' ).add( function () {
					this.fullCheck( api, lang );
					if ( mw.config.get( 'wgPageContentModel' ) === 'wikibase-property' ) {
						this.propertyParameterCheck( api, lang, entityId );
					}
				}.bind( this ) );
			}.bind( this ) );

			mw.hook( 'wikibase.statement.saved' ).add( function ( savedEntityId, statementId ) {
				mw.track( 'counter.MediaWiki.wikibase.quality.constraints.gadget.saveStatement' );
				this.snakCheck( api, lang, statementId );
			}.bind( this ) );
		}.bind( this ) );
	};

	SELF.prototype.fullCheck = function ( api, lang ) {
		var entity = this.getEntity(),
			entityIds = [ entity.getId() ];

		if ( typeof entity.getSubEntityIds === 'function' ) {
			entityIds = entityIds.concat( entity.getSubEntityIds() );
		}

		mw.track( 'counter.MediaWiki.wikibase.quality.constraints.gadget.loadEntity' );
		return this._fullCheckAllIds( api, lang, entityIds );
	};

	SELF.prototype._wbcheckconstraints = function ( api, lang, ids ) {
		return api.get( {
			action: 'wbcheckconstraints',
			format: 'json',
			formatversion: 2,
			uselang: lang,
			id: ids,
			status: this.config.CACHED_STATUSES
		} );
	};

	SELF.prototype._fullCheckAllIds = function ( api, lang, ids ) {
		var i,
			j = ids.length,
			chunk = this.config.WBCHECKCONSTRAINTS_MAX_ID_COUNT,
			promises = [];

		for ( i = 0; i < j; i += chunk ) {
			promises.push(
				this._wbcheckconstraints( api, lang, ids.slice( i, i + chunk ) )
			);
		}

		return $.when.apply( $, promises )
			.then( this._aggregateMultipleWbcheckconstraintsResponses.bind( this ) )
			.then( this._renderWbcheckconstraintsResult.bind( this ) )
			.promise();
	};

	/**
	 * Because of the way this is implemented it can not be done repeatedly for partial results
	 * (e.g. a wbcheckconstraints result for only some of the entities on a page)
	 *
	 * @param {Object} data Map of entity ids and constraint check information
	 */
	SELF.prototype._renderWbcheckconstraintsResult = function ( data ) {
		var self = this;

		$( '.wbqc-constraint-warning' ).remove();
		$( '.wikibase-statementgroupview .wikibase-statementview' )
			.each( function () {
				var entityId;
				for ( entityId in data ) {
					if ( !Object.prototype.hasOwnProperty.call( data, entityId ) ) {
						continue;
					}
					self._addReportsToStatement( data[ entityId ], $( this ) );
				}
			} );
	};

	SELF.prototype._aggregateMultipleWbcheckconstraintsResponses = function ( /* multiple responses */ ) {
		var responses = [].slice.call( arguments ),
			i = 0,
			responseCount = responses.length,
			entityConstraints = {};

		for ( i; i < responseCount; i++ ) {
			$.extend( entityConstraints, responses[ i ].wbcheckconstraints );
		}

		return entityConstraints;
	};

	SELF.prototype._getEntityDataByStatementId = function ( response, statementId ) {
		var entities = response.wbcheckconstraints,
			entity,
			property,
			properties,
			index;

		for ( entity in entities ) {

			if ( Object.prototype.hasOwnProperty.call( entities, entity ) ) {
				properties = entities[ entity ].claims;
				for ( property in properties ) {

					if ( Object.prototype.hasOwnProperty.call( properties, property ) ) {
						for ( index = 0; index < entities[ entity ].claims[ property ].length; index++ ) {

							if ( entities[ entity ].claims[ property ][ index ].id === statementId ) {
								return entities[ entity ];
							}

						}
					}

				}
			}

		}

		return null;
	};

	SELF.prototype.snakCheck = function ( api, lang, statementId ) {
		var isUpdated = false,
			statementClass = 'wikibase-statement-' + statementId.replace( /\$/, '\\$$' ),
			self = this;

		api.get( {
			action: 'wbcheckconstraints',
			format: 'json',
			formatversion: 2,
			uselang: lang,
			claimid: statementId,
			status: this.config.CACHED_STATUSES
		} ).then( function ( data ) {
			var entityData;
			if ( isUpdated ) {
				return;
			}
			entityData = self._getEntityDataByStatementId( data, statementId );
			if ( entityData !== null ) {
				self._addReportsToStatement( entityData,
					$( '.wikibase-statementgroupview .wikibase-statementview.' + statementClass )
				);
			}
		} );
		this.fullCheck( api, lang ).then( function () {
			isUpdated = true;
		} );
	};

	SELF.prototype.propertyParameterCheck = function ( api, lang, entityId ) {
		mw.track( 'counter.MediaWiki.wikibase.quality.constraints.gadget.loadProperty' );
		return api.get( {
			action: 'wbcheckconstraintparameters',
			format: 'json',
			formatversion: 2,
			uselang: lang,
			propertyid: entityId
		} ).then( function ( data ) {
			this._addParameterReports( data.wbcheckconstraintparameters[ entityId ] );
		}.bind( this ) );
	};

	/**
	 * Create a popup button according to the parameters and append it to the container.
	 *
	 * @param {jQuery} $content Content to be shown in the popup.
	 * @param {jQuery} $container The container to which the button is appended.
	 * @param {string} icon Identifier for an icon as provided by OOUI.
	 * @param {string} titleMessageKey The message key of the title attribute of the button.
	 * @param {string[]} [classes] Optional list of classes added to the button element
	 * @param {string} [flags] Optional custom flags the {@link OO.ui.PopupButtonWidget} can understand.
	 */
	SELF.prototype._buildPopup = function ( $content, $container, icon, titleMessageKey, classes, flags /* = '' */ ) {
		// eslint-disable-next-line mediawiki/class-doc
		var widget = new OO.ui.PopupButtonWidget( {
			icon: icon,
			// eslint-disable-next-line mediawiki/msg-doc
			title: mw.message( titleMessageKey ).text(),
			flags: flags || '',
			framed: false,
			classes: [ 'wbqc-reports-button', 'wikibase-snakview-indicator' ].concat( classes || [] ),
			$overlay: $container.parents( '.wikibase-statementview' ).first(),
			popup: {
				$content: $content,
				width: 400,
				padded: true,
				head: true,
				label: $content.find( '.wbqc-reports:first-child > .oo-ui-labelElement-label *' ).detach()
			}
		} );
		widget.popup.$element.css( 'z-index', 2 ); // prevent collision with rank selector and property grey field

		$container.append( widget.$element );
	};

	/**
	 * Get a message indicating the given cached status.
	 *
	 * @param {*} cached A value indicating the cached status.
	 * Typically an object with a `maximumAgeInSeconds` member.
	 * @return {string} HTML
	 */
	SELF.prototype._getCachedMessage = function ( cached ) {
		var maximumAgeInMinutes,
			maximumAgeInHours,
			maximumAgeInDays;
		if ( typeof cached === 'object' && cached.maximumAgeInSeconds ) {
			if ( cached.maximumAgeInSeconds < 60 * 90 ) {
				maximumAgeInMinutes = Math.ceil( cached.maximumAgeInSeconds / 60 );
				return mw.message( 'wbqc-cached-minutes' )
					.params( [ maximumAgeInMinutes ] )
					.escaped();
			}
			if ( cached.maximumAgeInSeconds < 60 * 60 * 36 ) {
				maximumAgeInHours = Math.ceil( cached.maximumAgeInSeconds / ( 60 * 60 ) );
				return mw.message( 'wbqc-cached-hours' )
					.params( [ maximumAgeInHours ] )
					.escaped();
			}
			maximumAgeInDays = Math.ceil( cached.maximumAgeInSeconds / ( 60 * 60 * 24 ) );
			return mw.message( 'wbqc-cached-days' )
				.params( [ maximumAgeInDays ] )
				.escaped();
		} else {
			return mw.message( 'wbqc-cached-generic' )
				.escaped();
		}
	};

	/**
	 * Build a panel for a single constraint check result.
	 *
	 * @param {Object} result The constraint check result.
	 * @return {wikibase.quality.constraints.ui.ConstraintReportPanel}
	 */
	SELF.prototype._buildReport = function ( result ) {
		var config = {
			status: result.status,
			constraint: result.constraint,
			message: result[ 'message-html' ],
			constraintClarification: result[ 'constraint-clarification' ]
		};
		if ( result.cached ) {
			config.ancillaryMessages = [ this._getCachedMessage( result.cached ) ];
		}
		return new wb.quality.constraints.ui.ConstraintReportPanel( config );
	};

	/**
	 * Build a panel for a single constraint parameter check result.
	 * This is to `wbcheckconstraintparameters` what {@link this._buildReport} is to `wbcheckconstraints`.
	 *
	 * @param {Object} problem The constraint parameter check result.
	 * @return {OO.ui.PanelLayout}
	 */
	SELF.prototype._buildParameterReport = function ( problem ) {
		var $report, $heading, $body;

		$report = $( '<div>' )
			.addClass( 'wbqc-parameter-report' );
		$heading = $( '<h4>' ); // empty for now
		$report.append( $heading );
		$body = $( '<p>' )
			.html( problem[ 'message-html' ] );
		$report.append( $body );

		return new OO.ui.PanelLayout( {
			expanded: false,
			$content: $report
		} );
	};

	/**
	 * Build a list of constraint reports from a list of panels,
	 * but return it only if the list is nonempty and contains uncollapsed items.
	 *
	 * @param {wikibase.quality.constraints.ui.ConstraintReportPanel[]} reports
	 * A list of individual report panels as returned by {@link this._buildReport}.
	 * @return {wikibase.quality.constraints.ui.ConstraintReportList|null}
	 * The list if it contains at least one uncollapsed panel,
	 * otherwise null.
	 */
	SELF.prototype._buildReportList = function ( reports ) {
		var list = wikibase.quality.constraints.ui.ConstraintReportList.static.fromPanels(
			reports,
			{
				statuses: [
					{
						status: 'violation',
						label: mw.message( 'wbqc-issues-short' ).text()
					},
					{
						status: 'warning',
						label: mw.message( 'wbqc-potentialissues-short' ).text()
					},
					{
						status: 'suggestion',
						label: mw.message( 'wbqc-suggestions-short' ).text()
					},
					{
						status: 'bad-parameters',
						label: mw.message( 'wbqc-parameterissues-short' ).text(),
						subheading: mw.message( 'wbqc-parameterissues-long' ).text(),
						collapsed: true
					}
				],
				expanded: false // expanded: true does not work within a popup
			}
		);
		if (
			// list isn't empty...
			list.items.length > 0 &&
			// ...and doesn't only contain collapsed items either
			list.items[ 0 ].status !== 'bad-parameters'
		) {
			return list;
		} else {
			return null;
		}
	};

	/**
	 * Extract the constraint check results for a certain statement from the full API response.
	 *
	 * @param {Object} entityData The constraint check results for a single entity.
	 * @param {string} propertyId The serialization of the property ID of the statement.
	 * @param {string} statementId The ID of the statement.
	 * @return {Object|null} An object containing lists of constraint check results,
	 * or null if the results could not be extracted.
	 */
	SELF.prototype._extractResultsForStatement = function ( entityData, propertyId, statementId ) {
		var statements, index, statement, results = {}, qualifierPID, referenceIndex, referenceHash, referencePID;
		if ( 'claims' in entityData ) {
			// API v2 format
			statements = entityData.claims[ propertyId ];
			if ( statements === undefined ) {
				return null;
			}
			for ( index in statements ) {
				if ( statements[ index ].id === statementId ) {
					statement = statements[ index ];
					break;
				}
			}
			if ( statement === undefined ) {
				return null;
			}

			results.mainsnak = statement.mainsnak.results;

			results.qualifiers = [];
			for ( qualifierPID in statement.qualifiers ) {
				for ( index in statement.qualifiers[ qualifierPID ] ) {
					results.qualifiers.push(
						statement.qualifiers[ qualifierPID ][ index ]
					);
				}
			}

			results.references = [];
			for ( referenceIndex in statement.references ) {
				referenceHash = statement.references[ referenceIndex ].hash;
				if ( !( referenceHash in results.references ) ) {
					results.references[ referenceHash ] = [];
				}
				for ( referencePID in statement.references[ referenceIndex ].snaks ) {
					for ( index in statement.references[ referenceIndex ].snaks[ referencePID ] ) {
						results.references[ referenceHash ].push(
							statement.references[ referenceIndex ].snaks[ referencePID ][ index ]
						);
					}
				}
			}

			return results;
		} else {
			// API v1 format
			if ( propertyId in entityData && statementId in entityData[ propertyId ] ) {
				return {
					mainsnak: entityData[ propertyId ][ statementId ]
				};
			} else {
				return null;
			}
		}
	};

	/**
	 * Add a popup for the given results to the given snak,
	 * if there are any results to display.
	 *
	 * @param {Array} results A list of constraint check results.
	 * @param {jQuery} $snak The snak to which the results apply.
	 * @return {boolean} Whether any results were added to the snak or not.
	 */
	SELF.prototype._addResultsToSnak = function ( results, $snak ) {
		var reports = results.map( this._buildReport.bind( this ) ),
			list = this._buildReportList( reports ),
			icon,
			titleMessageKey;

		if ( list !== null ) {
			switch ( list.items[ 0 ].status ) {
				case 'violation':
					icon = 'mandatory-constraint-violation';
					titleMessageKey = 'wbqc-issues-long';
					break;
				case 'warning':
					icon = 'non-mandatory-constraint-violation';
					titleMessageKey = 'wbqc-potentialissues-long';
					break;
				case 'suggestion':
					icon = 'suggestion-constraint-violation';
					titleMessageKey = 'wbqc-suggestions-long';
					break;
				default:
					throw new Error( 'unexpected status ' + list.items[ 0 ].status );
			}

			this._buildPopup(
				list.$element,
				$snak.find( '.wikibase-snakview-indicators' ),
				icon,
				titleMessageKey,
				[ 'wbqc-constraint-warning' ]
			);
			return true;
		} else {
			return false;
		}
	};

	/**
	 * Add constraint check results to a certain statement.
	 *
	 * @param {Object} entityData The constraint check results for a single entity.
	 * @param {jQuery} $statement The statement.
	 */
	SELF.prototype._addReportsToStatement = function ( entityData, $statement ) {
		var match = $statement[ 0 ].className.match(
				/\bwikibase-statement-([^\s$]+\$[\dA-F-]+)\b/i
			),
			statementId = match && match[ 1 ],
			propertyId = $statement.parents( '.wikibase-statementgroupview' ).data( 'property-id' ),
			results = this._extractResultsForStatement( entityData, propertyId, statementId ),
			index,
			qualifier,
			hash,
			reference,
			hasResults;

		if ( results === null ) {
			return;
		}

		this._addResultsToSnak(
			results.mainsnak,
			$statement.find( '.wikibase-statementview-mainsnak' )
		);

		for ( index in results.qualifiers ) {
			qualifier = results.qualifiers[ index ];
			this._addResultsToSnak(
				qualifier.results,
				$statement.find(
					'.wikibase-statementview-qualifiers ' +
					'.wikibase-snakview-' + qualifier.hash
				)
			);
		}

		for ( hash in results.references ) {
			for ( index in results.references[ hash ] ) {
				reference = results.references[ hash ][ index ];
				hasResults = this._addResultsToSnak(
					reference.results,
					$statement.find(
						'.wikibase-statementview-references ' +
						'.wikibase-referenceview-' + hash + ' ' +
						'.wikibase-snakview-' + reference.hash
					)
				);
				if ( hasResults ) {
					mw.hook( 'wikibase.entityPage.entityView.rendered' ).add( function () {
						var $referenceToggler = $statement
							.find( '.wikibase-statementview-references-heading .ui-toggler-toggle-collapsed' );
						if ( $referenceToggler.length > 0 ) {
							$referenceToggler.data( 'toggler' ).toggle();
						}
					} );
				}
			}
		}
	};

	/**
	 * Add constraint parameter check results for all constraint statements listed in the reports.
	 *
	 * @param {Object} parameterReports A map from constraint ID (i.e., constraint statement ID)
	 * to results for that parameter (overall status and list of problems).
	 */
	SELF.prototype._addParameterReports = function ( parameterReports ) {
		var constraintId,
			status,
			problems,
			reports,
			list,
			$snak;

		for ( constraintId in parameterReports ) {
			status = parameterReports[ constraintId ].status;
			if ( status === 'okay' ) {
				continue;
			}

			problems = parameterReports[ constraintId ].problems;
			reports = problems.map( this._buildParameterReport.bind( this ) );

			list = new wikibase.quality.constraints.ui.ConstraintReportList( {
				items: [
					new wikibase.quality.constraints.ui.ConstraintReportGroup( {
						items: reports,
						label: mw.message( 'wbqc-badparameters-short' ).text()
					} )
				],
				expanded: false // expanded: true does not work within a popup
			} );

			$snak = $( '.wikibase-statement-' + constraintId.replace( /\$/g, '\\$' ) +
				' .wikibase-statementview-mainsnak .wikibase-snakview' );
			this._buildPopup(
				list.$element,
				$snak.find( '.wikibase-snakview-indicators' ),
				'alert',
				'wbqc-badparameters-long',
				[ 'wbqc-parameter-warning' ],
				'warning'
			);
		}
	};

	if ( typeof exports === 'undefined' ) {
		( new SELF() ).defaultBehavior();
	}

	return SELF;

}( mediaWiki, wikibase, jQuery, OO ) );
