describe( 'wikibase.quality.constraints.gadget', function () {
	var expect = require( 'unexpected' ).clone(),
		sinon = require( 'sinon' ),
		Gadget,
		loadedEntity;

	beforeEach( function () {
		// ensure the module, containing immediately invoked code, is loaded repeatedly
		delete require.cache[ require.resolve( 'wikibase.quality.constraints.gadget' ) ];

		global.mediaWiki = sinon.stub();
		global.wikibase = sinon.stub();
		global.jQuery = sinon.stub();
		global.OO = sinon.stub();

		// poor man's Promise.all()
		global.jQuery.when = function () {
			var promises = arguments,
				length = promises.length,
				results = [],
				i;
			for ( i = 0; i < length; i++ ) {
				promises[ i ].then( function ( data ) {
					results.push( data );
				} );
			}
			return {
				then: function ( func ) {
					var thenResult = func.apply( {}, results );
					return {
						then: function ( func2 ) {
							func2( thenResult );
							return {
								promise: sinon.spy()
							};
						}
					};
				}
			};
		};
		// poor man's Object.assign()
		global.jQuery.extend = function () {
			var objects = Array.from( arguments ),
				target = objects.shift(),
				i,
				name;
			for ( i = 0; i < objects.length; i++ ) {
				for ( name in objects[ i ] ) {
					target[ name ] = objects[ i ][ name ];
				}
			}
			return target;
		};

		loadedEntity = {};

		global.wikibase.EntityInitializer = {
			newFromEntityLoadedHook: function () {
				return {
					getEntity: function () {
						return {
							done: sinon.stub().yields( loadedEntity )
						};
					}
				};
			}
		};

		Gadget = require( 'wikibase.quality.constraints.gadget' );
	} );

	it( 'exports an invokable module', function () {
		expect( typeof Gadget, 'to equal', 'function' );
	} );

	describe( 'config', function () {
		it( 'has default values', function () {
			var gadget = new Gadget();
			expect( gadget.config.CACHED_STATUSES, 'to equal', 'violation|warning|suggestion|bad-parameters' );
			expect( gadget.config.WBCHECKCONSTRAINTS_MAX_ID_COUNT, 'to equal', 50 );
		} );

		it( 'can be overwritten by constructor parameter', function () {
			var gadget = new Gadget( { WBCHECKCONSTRAINTS_MAX_ID_COUNT: 3 } );
			expect( gadget.config.WBCHECKCONSTRAINTS_MAX_ID_COUNT, 'to equal', 3 );
		} );
	} );

	describe( 'default behavior', function () {
		it( 'gets entity id from wbEntityId', function () {
			var gadget = new Gadget();

			global.mediaWiki.config = sinon.stub();
			global.mediaWiki.config.get = sinon.stub();

			gadget.defaultBehavior();

			expect( global.mediaWiki.config.get.withArgs( 'wbEntityId' ).calledOnce, 'to be true' );
		} );

		it( 'checks if wbIsEditView true', function () {
			var gadget = new Gadget();

			global.mediaWiki.config = sinon.stub();
			global.mediaWiki.config.get = sinon.stub();
			global.mediaWiki.config.get.withArgs( 'wbEntityId' ).returns( 'Q42' );

			gadget.defaultBehavior();

			expect( global.mediaWiki.config.get.withArgs( 'wbIsEditView' ).calledOnce, 'to be true' );
		} );

		it( 'sets entity from newFromEntityLoadedHook', function () {
			var gadget = new Gadget();

			gadget.fullCheck = sinon.stub();
			global.mediaWiki.config = sinon.stub();
			global.mediaWiki.config.get = sinon.stub();
			global.mediaWiki.config.get.withArgs( 'wbEntityId' ).returns( 'Q42' );
			global.mediaWiki.config.get.withArgs( 'wbIsEditView' ).returns( true );
			global.mediaWiki.loader = sinon.stub();
			global.mediaWiki.loader.using = sinon.stub();
			global.mediaWiki.loader.using.returns( { done: sinon.stub().yields() } );
			global.mediaWiki.loader.getState = sinon.stub();
			global.mediaWiki.loader.getState.withArgs( 'wikibase.quality.constraints.gadget' ).returns( 'executing' );
			global.mediaWiki.Api = sinon.stub();
			global.mediaWiki.Api.prototype.get = sinon.stub().returns( {
				then: sinon.stub().returns( {
					then: sinon.stub()
				} )
			} );

			global.mediaWiki.hook = sinon.stub();
			loadedEntity = {
				getId: sinon.stub().returns( sinon.stub() )
			};
			global.mediaWiki.hook.returns( { add: sinon.stub() } );

			gadget.defaultBehavior();

			expect( gadget.getEntity(), 'to equal', loadedEntity );
		} );

		it( 'invokes mw loader and resumes once it is ready', function () {
			var gadget = new Gadget(),
				done = sinon.stub();

			global.mediaWiki.config = sinon.stub();
			global.mediaWiki.config.get = sinon.stub();
			global.mediaWiki.config.get.withArgs( 'wbEntityId' ).returns( 'Q42' );
			global.mediaWiki.config.get.withArgs( 'wbIsEditView' ).returns( true );
			global.mediaWiki.loader = sinon.stub();
			global.mediaWiki.loader.using = sinon.stub();
			global.mediaWiki.loader.using.returns( { done: done } );

			gadget.defaultBehavior();

			expect( global.mediaWiki.loader.using.calledOnce, 'to be true' );
			expect( done.calledOnce, 'to be true' );
		} );

		it( 'runs a fullCheck once mw loader is done and entityView.rendered fires', function () {
			var gadget = new Gadget(),
				statementSavedSpy = sinon.spy(),
				entityViewRenderedSpy = sinon.stub(),
				wgUserLanguage = 'de',
				wbEntityId = 'Q42';

			global.mediaWiki.config = sinon.stub();
			global.mediaWiki.config.get = sinon.stub();
			global.mediaWiki.config.get.withArgs( 'wbEntityId' ).returns( wbEntityId );
			global.mediaWiki.config.get.withArgs( 'wbIsEditView' ).returns( true );
			global.mediaWiki.loader = sinon.stub();
			global.mediaWiki.loader.using = sinon.stub();
			global.mediaWiki.loader.using.returns( { done: sinon.stub().yields() } );

			global.mediaWiki.loader.getState = sinon.stub();
			global.mediaWiki.loader.getState.withArgs( 'wikibase.quality.constraints.gadget' ).returns( 'executing' );
			global.mediaWiki.Api = sinon.stub();
			global.mediaWiki.Api.prototype.get = sinon.stub().returns( {
				then: sinon.stub().returns( {
					then: sinon.stub()
				} )
			} );
			global.mediaWiki.config.get.withArgs( 'wgUserLanguage' ).returns( wgUserLanguage );

			global.mediaWiki.hook = sinon.stub();
			loadedEntity = {
				getId: sinon.stub().returns( wbEntityId )
			};
			global.mediaWiki.hook.withArgs( 'wikibase.statement.saved' ).returns( { add: statementSavedSpy } );
			global.mediaWiki.hook.withArgs( 'wikibase.entityPage.entityView.rendered' )
				.returns( { add: entityViewRenderedSpy.yields() } );

			global.mediaWiki.track = sinon.spy();

			gadget._renderWbcheckconstraintsResult = sinon.stub();

			gadget.defaultBehavior();

			expect( entityViewRenderedSpy.calledOnce, 'to be true' );
			sinon.assert.calledWith(
				global.mediaWiki.track,
				'counter.MediaWiki.wikibase.quality.constraints.gadget.loadEntity'
			);
			sinon.assert.calledWith( global.mediaWiki.Api.prototype.get, {
				action: 'wbcheckconstraints',
				format: 'json',
				formatversion: 2,
				uselang: wgUserLanguage,
				id: [ wbEntityId ], // in a real request the mw.Api would format this with pipes
				status: gadget.config.CACHED_STATUSES
			} );
			sinon.assert.callCount( gadget._renderWbcheckconstraintsResult, 1 );
			sinon.assert.calledWith( global.mediaWiki.hook, 'wikibase.statement.saved' );
			expect( statementSavedSpy.calledOnce, 'to be true' );
		} );
	} );

	describe( 'setting and getting an entity', function () {
		it( 'gets the same entity as set', function () {
			var gadget = new Gadget(),
				entity = sinon.stub();
			gadget.setEntity( entity );
			expect( gadget.getEntity(), 'to equal', entity );
		} );
	} );

	describe( 'fullCheck', function () {
		it( 'tracks usage', function () {
			var gadget = new Gadget(),
				lang = 'fr',
				entityId = 'Q42',
				api = {
					get: sinon.stub().returns( {
						then: sinon.stub().returns( {
							then: sinon.stub()
						} )
					} )
				};

			global.mediaWiki.track = sinon.spy();

			gadget._renderWbcheckconstraintsResult = sinon.spy();
			gadget.setEntity( { getId: sinon.stub().returns( entityId ) } );
			gadget.fullCheck( api, lang );

			sinon.assert.calledWith( global.mediaWiki.track, 'counter.MediaWiki.wikibase.quality.constraints.gadget.loadEntity' );
		} );

		it( 'calls api with correct parameters', function () {
			var gadget = new Gadget(),
				lang = 'fr',
				entityId = 'Q42',
				api = {
					get: sinon.stub().returns( {
						then: sinon.stub().returns( {
							then: sinon.stub()
						} )
					} )
				};

			global.mediaWiki.track = sinon.spy();

			gadget._renderWbcheckconstraintsResult = sinon.spy();

			gadget.setEntity( { getId: sinon.stub().returns( entityId ) } );
			gadget.fullCheck( api, lang );

			sinon.assert.calledWith( api.get, {
				action: 'wbcheckconstraints',
				format: 'json',
				formatversion: 2,
				uselang: 'fr',
				id: [ entityId ],
				status: gadget.config.CACHED_STATUSES
			} );
		} );

		it( 'uses api response to update DOM statements', function () {
			var gadget = new Gadget(),
				lang = 'fr',
				entityId = 'Q42',
				api = {
					get: sinon.stub()
				},
				responseData = {
					wbcheckconstraints: {
						Q42: {
							claims: {
								P9: [ {
									id: 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2',
									mainsnak: {
										hash: '261be448fb9ca79fd5e3fb45e7b810c5d33c2e4d',
										results: [ {
											status: 'warning',
											property: 'P9',
											constraint: {
												id: 'P9$197313ce-4014-c893-e2c4-5eb1f9347945',
												type: 'Q1283',
												typeLabel: 'MultiValueConstraintItem',
												link: 'https://test.wikidata.org/wiki/Property:P9#P9$197313ce-4014-c893-e2c4-5eb1f9347945',
												discussLink: 'https://test.wikidata.org/wiki/Property_talk:P9'
											},
											'message-html': 'This property should contain multiple values.',
											claim: 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2'
										} ]
									}
								} ]
							}
						}
					}, success: 1
				},
				preExistingReportButtonsRemovedSpy = sinon.spy();

			global.mediaWiki.track = sinon.spy();
			api.get.withArgs( {
				action: 'wbcheckconstraints',
				format: 'json',
				formatversion: 2,
				uselang: 'fr',
				id: [ entityId ],
				status: gadget.config.CACHED_STATUSES
			} ).returns( {
				then: sinon.stub().yields( responseData )
			} );
			global.jQuery.withArgs( '.wbqc-constraint-warning' ).returns( {
				remove: preExistingReportButtonsRemovedSpy
			} );
			global.jQuery.withArgs( '.wikibase-statementgroupview .wikibase-statementview' ).returns( {
				each: sinon.stub().yields()
			} );
			gadget._addReportsToStatement = sinon.spy();

			gadget.setEntity( { getId: sinon.stub().returns( entityId ) } );
			gadget.fullCheck( api, lang );

			sinon.assert.called( preExistingReportButtonsRemovedSpy );
			sinon.assert.calledWith( gadget._addReportsToStatement, responseData.wbcheckconstraints.Q42 );
		} );
	} );

	describe( '_fullCheckAllIds', function () {
		it( 'chunks requests', function () {
			var gadget = new Gadget( { WBCHECKCONSTRAINTS_MAX_ID_COUNT: 2 } ),
				api = sinon.stub(),
				lang = 'fr';

			gadget._wbcheckconstraints = sinon.stub();
			gadget._wbcheckconstraints.returns( { then: sinon.stub() } );
			gadget._aggregateMultipleWbcheckconstraintsResponses = sinon.spy();
			gadget._renderWbcheckconstraintsResult = sinon.spy();

			gadget._fullCheckAllIds( api, lang, [ 'L2', 'L2-F1', 'L2-F2', 'L2-F3', 'L2-S1', 'L2-S2', 'L2-F3' ] );

			sinon.assert.callCount( gadget._wbcheckconstraints, 4 );
			sinon.assert.calledWith( gadget._wbcheckconstraints, api, lang, [ 'L2', 'L2-F1' ] );
			sinon.assert.calledWith( gadget._wbcheckconstraints, api, lang, [ 'L2-F2', 'L2-F3' ] );
			sinon.assert.calledWith( gadget._wbcheckconstraints, api, lang, [ 'L2-S1', 'L2-S2' ] );
			sinon.assert.calledWith( gadget._wbcheckconstraints, api, lang, [ 'L2-F3' ] );

			sinon.assert.callCount( gadget._aggregateMultipleWbcheckconstraintsResponses, 1 );
			sinon.assert.callCount( gadget._renderWbcheckconstraintsResult, 1 );
		} );
	} );

	describe( '_aggregateMultipleWbcheckconstraintsResponses', function () {
		it( 'can combine multiple responses\' entity information', function () {
			var gadget = new Gadget(),
				responseOne = {
					wbcheckconstraints: {
						L2: {
							data: 'here'
						}
					}
				},
				responseTwo = {
					wbcheckconstraints: {
						'L2-F1': {
							data: 'more of it'
						}
					}
				},
				combinedResponse;

			combinedResponse = gadget._aggregateMultipleWbcheckconstraintsResponses( responseOne, responseTwo );

			expect( combinedResponse.L2, 'to equal', responseOne.wbcheckconstraints.L2 );
			expect( combinedResponse[ 'L2-F1' ], 'to equal', responseTwo.wbcheckconstraints[ 'L2-F1' ] );
		} );
	} );

	describe( '_addReportsToStatement', function () {
		var entityData = {
				claims: {
					P9: [ {
						id: 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2',
						mainsnak: {
							hash: '261be448fb9ca79fd5e3fb45e7b810c5d33c2e4d',
							results: [ {
								status: 'warning',
								property: 'P9',
								constraint: {
									id: 'P9$197313ce-4014-c893-e2c4-5eb1f9347945',
									type: 'Q1283',
									typeLabel: 'MultiValueConstraintItem',
									link: 'https://test.wikidata.org/wiki/Property:P9#P9$197313ce-4014-c893-e2c4-5eb1f9347945',
									discussLink: 'https://test.wikidata.org/wiki/Property_talk:P9'
								},
								'message-html': 'This property should contain multiple values.',
								claim: 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2'
							} ]
						},
						qualifiers: {
							P9: [ {
								hash: 'ab294fd4a345d913a4c96a91b184e563f9431847',
								results: [ {
									status: 'warning',
									property: 'P9',
									constraint: {
										id: 'P9$197313ce-4014-c893-e2c4-5eb1f9347945',
										type: 'Q1283',
										typeLabel: 'MultiValueConstraintItem',
										link: 'https://test.wikidata.org/wiki/Property:P9#P9$197313ce-4014-c893-e2c4-5eb1f9347945',
										discussLink: 'https://test.wikidata.org/wiki/Property_talk:P9'
									},
									'message-html': 'This property should contain multiple values.'
								} ]
							} ]
						},
						references: [
							{
								hash: 'ec294fd4a345d913a4c96a91b184e563f9431832',
								snaks: {
									P9: [ {
										hash: '261be448fb9ca79fd5e3fb45e7b810c5d33c2e4d',
										results: [ {
											status: 'warning',
											property: 'P9',
											constraint: {
												id: 'P9$197313ce-4014-c893-e2c4-5eb1f9347945',
												type: 'Q1283',
												typeLabel: 'MultiValueConstraintItem',
												link: 'https://test.wikidata.org/wiki/Property:P9#P9$197313ce-4014-c893-e2c4-5eb1f9347945',
												discussLink: 'https://test.wikidata.org/wiki/Property_talk:P9'
											},
											'message-html': 'This property should contain multiple values.'
										} ]
									} ]
								}
							}
						]
					} ]
				}
			},
			dataStub = sinon.stub().withArgs( 'property-id' ).returns( 'P9' ),
			$statement = {
				0: {
					className: 'wikibase-statement-Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2'
				},
				parents: sinon.stub().returns( {
					data: dataStub
				} )
			};

		it( 'extracts result for statement with property id and statement id', function () {
			var gadget = new Gadget();

			gadget._extractResultsForStatement = sinon.stub();
			gadget._extractResultsForStatement.returns( null );

			gadget._addReportsToStatement( entityData, $statement );

			sinon.assert.calledWith(
				gadget._extractResultsForStatement,
				entityData,
				'P9',
				'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2'
			);
		} );

		it( 'adds results to main snak', function () {
			var gadget = new Gadget(),
				$mainsnak = {};

			gadget._extractResultsForStatement = sinon.stub()
				.withArgs( entityData, 'P9', 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2' )
				.returns( entityData.claims.P9[ 0 ] );
			$statement.find = sinon.stub().withArgs( '.wikibase-statementview-mainsnak' ).returns( $mainsnak );
			gadget._addResultsToSnak = sinon.spy();

			gadget._addReportsToStatement( entityData, $statement );

			sinon.assert.calledWith(
				gadget._addResultsToSnak,
				entityData.claims.P9[ 0 ].mainsnak,
				$mainsnak
			);
		} );

		it( 'adds results to qualifiers', function () {
			var gadget = new Gadget(),
				$qualifierSnak = {};

			gadget._extractResultsForStatement = sinon.stub()
				.withArgs( entityData, 'P9', 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2' )
				.returns( { // gets reformatted by _extractResultsForStatement
					qualifiers: entityData.claims.P9[ 0 ].qualifiers.P9
				} );
			$statement.find = sinon.stub()
				.withArgs( '.wikibase-statementview-qualifiers .wikibase-snakview-ab294fd4a345d913a4c96a91b184e563f9431847' )
				.returns( $qualifierSnak );
			gadget._addResultsToSnak = sinon.spy();

			gadget._addReportsToStatement( entityData, $statement );

			sinon.assert.calledWith(
				gadget._addResultsToSnak,
				entityData.claims.P9[ 0 ].qualifiers.P9[ 0 ].results,
				$qualifierSnak
			);
		} );

		it( 'adds results to references', function () {
			var gadget = new Gadget(),
				$referenceSnak = {},
				entityViewrenderedHook = sinon.stub().yields(),
				togglerToggledSpy = sinon.spy();

			gadget._extractResultsForStatement = sinon.stub()
				.withArgs( entityData, 'P9', 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2' )
				.returns( { // gets reformatted by _extractResultsForStatement
					references: {
						ec294fd4a345d913a4c96a91b184e563f9431832: [
							entityData.claims.P9[ 0 ].references[ 0 ].snaks.P9[ 0 ]
						]
					}
				} );
			$statement.find = sinon.stub();
			$statement.find.withArgs(
				'.wikibase-statementview-references ' +
				'.wikibase-referenceview-ec294fd4a345d913a4c96a91b184e563f9431832 ' +
				'.wikibase-snakview-261be448fb9ca79fd5e3fb45e7b810c5d33c2e4d'
			).returns( $referenceSnak );
			gadget._addResultsToSnak = sinon.stub().returns( true );
			global.mediaWiki.hook = sinon.stub();
			global.mediaWiki.hook.withArgs( 'wikibase.entityPage.entityView.rendered' ).returns( {
				add: entityViewrenderedHook
			} );
			$statement.find.withArgs( '.wikibase-statementview-references-heading .ui-toggler-toggle-collapsed' )
				.returns( {
					length: 1,
					data: sinon.stub().withArgs( 'toggler' ).returns( {
						toggle: togglerToggledSpy
					} )
				} );

			gadget._addReportsToStatement( entityData, $statement );

			sinon.assert.calledWith(
				gadget._addResultsToSnak,
				entityData.claims.P9[ 0 ].references[ 0 ].snaks.P9[ 0 ].results,
				$referenceSnak
			);
			sinon.assert.calledOnce( togglerToggledSpy );
		} );
	} );

	describe( '_extractResultsForStatement', function () {
		it( 'finds constraint violation result in entity data', function () {
			var gadget = new Gadget(),
				entityData = {
					claims: {
						P9: [ {
							id: 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2',
							mainsnak: {
								hash: '261be448fb9ca79fd5e3fb45e7b810c5d33c2e4d',
								results: [ {
									status: 'warning',
									property: 'P9',
									constraint: {
										id: 'P9$197313ce-4014-c893-e2c4-5eb1f9347945',
										type: 'Q1283',
										typeLabel: 'MultiValueConstraintItem',
										link: 'https://test.wikidata.org/wiki/Property:P9#P9$197313ce-4014-c893-e2c4-5eb1f9347945',
										discussLink: 'https://test.wikidata.org/wiki/Property_talk:P9'
									},
									'message-html': 'This property should contain multiple values.',
									claim: 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2'
								} ]
							}
						} ]
					}
				},
				result;

			result = gadget._extractResultsForStatement( entityData, 'P9', 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2' );

			expect( result.qualifiers, 'to equal', [] );
			expect( result.references, 'to equal', [] );
			expect( result.mainsnak, 'to equal', entityData.claims.P9[ 0 ].mainsnak.results );
		} );
	} );

	describe( '_getEntityDataByStatementId', function () {
		it( 'extracts the entity data when the statementId exists', function () {
			var gadget = new Gadget(),
				entityData = {
					claims: {
						P9: [ {
							id: 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2'
						} ]
					}
				},
				result;
			result = gadget._getEntityDataByStatementId( { wbcheckconstraints: { Q42: entityData } },
				'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2' );

			expect( result, 'to equal', entityData );
		} );

		it( 'returns null when the statementId isn\'t present', function () {
			var gadget = new Gadget(),
				entityData = {
					claims: {
						P9: [ {
							id: 'noid'
						} ]
					}
				},
				result;
			result = gadget._getEntityDataByStatementId( { wbcheckconstraints: { Q42: entityData } },
				'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2' );

			expect( result, 'to equal', null );
		} );
	} );

	describe( 'snackCheck', function () {
		it( 'runs a full check', function () {
			var gadget = new Gadget(),
				api = sinon.stub(),
				lang = sinon.stub(),
				statementId = sinon.stub();

			statementId.replace = sinon.stub();
			api.get = sinon.stub().returns( { then: sinon.stub() } );
			gadget.fullCheck = sinon.stub().returns( { then: sinon.stub() } );

			gadget.snakCheck( api, lang, statementId );
			sinon.assert.calledOnce( gadget.fullCheck );
		} );

		it( 'adds reports to statement from response', function () {
			var gadget = new Gadget(),
				api = sinon.stub(),
				lang = sinon.stub(),
				statementId = 'Q42$ae34ea61-4b81-db2b-4220-28f115fff19b',
				responseData,
				entityData;

			entityData = { claims: { P3: [
				{
					id: 'Q42$ae34ea61-4b81-db2b-4220-28f115fff19b',
					mainsnak: {
						hash: '5929cec551a5f2e1dd9e07b5531cb01948c06142',
						results: []
					}
				}
			] } };

			responseData = { wbcheckconstraints: { Q42: entityData } };

			api.get = sinon.stub().returns( { then: sinon.stub().yields( responseData ) } );
			gadget.fullCheck = sinon.stub().returns( { then: sinon.stub() } );
			gadget._addReportsToStatement = sinon.spy();

			gadget.snakCheck( api, lang, statementId );
			sinon.assert.calledWith( gadget._addReportsToStatement, entityData );
		} );

		it( 'calls api with statement id', function () {
			var gadget = new Gadget(),
				api = sinon.stub(),
				lang = 'de',
				statementId = 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2';

			api.get = sinon.stub().returns( { then: sinon.stub() } );
			gadget.fullCheck = sinon.stub().returns( { then: sinon.stub() } );

			gadget.snakCheck( api, lang, statementId );

			sinon.assert.calledWith( api.get, {
				action: 'wbcheckconstraints',
				format: 'json',
				formatversion: 2,
				uselang: 'de',
				claimid: 'Q42$2c9c5e39-4d4c-23f0-5bcb-b92615bf7aa2',
				status: gadget.config.CACHED_STATUSES
			} );
		} );
	} );

} );
