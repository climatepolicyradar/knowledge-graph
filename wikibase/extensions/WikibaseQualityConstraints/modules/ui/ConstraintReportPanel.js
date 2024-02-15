( function ( mw, wb ) {
	'use strict';

	/**
	 * A panel for a single constraint report result.
	 *
	 *     @example
	 *     var report = new wikibase.quality.constraints.ui.ConstraintReportPanel( {
	 *         status: 'compliance',
	 *         constraint: {
	 *             type: 'Q1',
	 *             typeLabel: 'my constraint',
	 *             link: 'http://example.com/my-constraint',
	 *             discussLink: 'http://example.com/Property_talk:P2',
	 *         },
	 *         message: 'everything okay',
	 *         expanded: false,
	 *         framed: true
	 *     } );
	 *     $( 'body' ).append( report.$element );
	 *
	 * @class
	 * @extends OO.ui.PanelLayout
	 *
	 * @constructor
	 * @param {Object} config Configuration options
	 * @cfg {string} status The status of the report, e.g. 'violation' or 'compliance'.
	 * @cfg {Object} constraint The constraint of the report, as returned by the wbcheckconstraints API.
	 * @cfg {string} [message=''] The message (HTML) of the report, if present.
	 * @cfg {string} [constraintClarification] The constraint clarification (plain text) of the report, if any.
	 * @cfg {string[]} [ancillaryMessages=[]] Additional messages (HTML) attached to the report, if any.
	 * @cfg {jQuery} [$heading] The heading element of the panel. Should not yet contain the heading links.
	 * @cfg {jQuery} [$headingLinks] The container for the links in the heading of the panel. Should not yet contain the help link or the discussion link.
	 * @cfg {jQuery} [$helpLink] The help link for the heading.
	 * @cfg {jQuery} [$discussLink] The link to the discussion page.
	 * @cfg {jQuery} [$message] The message paragraph of the panel.
	 * @cfg {jQuery} [$constraintClarification] The constraint clarification paragraph of the panel.
	 * @cfg {jQuery} [$ancillaryMessages] The container of the additional messages.
	 */
	wb.quality.constraints.ui.ConstraintReportPanel = function WBQCConstraintReportPanel( config ) {
		// Configuration initialization
		config = $.extend( {
			message: '',
			ancillaryMessages: []
		}, config );
		config.expanded = false;

		// Parent constructor
		wb.quality.constraints.ui.ConstraintReportPanel.parent.call( this, config );

		// Mixin constructors
		// (none)

		// Properties
		this.status = config.status;
		this.constraint = config.constraint;
		this.$heading = config.$heading || $( '<h4 class="wbqc-report-heading">' ).append(
			$( '<a>' )
				.text( this.constraint.typeLabel )
				.attr( 'href', this.constraint.link )
				.attr( 'target', '_blank' )
		);
		this.$headingLinks = config.$headingLinks || $( '<small class="wbqc-report-heading-links">' );
		this.$helpLink = config.$helpLink || $( '<a class="wbqc-constraint-type-help">' )
			.text( mw.message( 'wbqc-constrainttypehelp-short' ).text() )
			.attr( 'title', mw.message( 'wbqc-constrainttypehelp-long' ).text() )
			.attr(
				'href',
				'https://www.wikidata.org/wiki/Special:MyLanguage/Help:Property_constraints_portal/' + this.constraint.type
			)
			.attr( 'target', '_blank' );
		this.$discussLink = config.$discussLink || $( '<a class="wbqc-constraint-discuss">' )
			.text( mw.message( 'wbqc-constraintdiscuss-short' ).text() )
			.attr( 'title', mw.message( 'wbqc-constraintdiscuss-long' ).text() )
			.attr( 'href', this.constraint.discussLink )
			.attr( 'target', '_blank' );
		this.message = config.message;
		this.$message = config.$message || $( '<p>' ).html( this.message );
		this.constraintClarification = config.constraintClarification;
		this.$constraintClarification = config.$constraintClarification ||
			( this.constraintClarification && $( '<p>' ).text( this.constraintClarification ) );
		this.ancillaryMessages = config.ancillaryMessages;
		this.$ancillaryMessages = config.$ancillaryMessages || $( '<div class="wbqc-ancillary-messages">' ).append(
			this.ancillaryMessages.map( function ( ancillaryMessage ) {
				return $( '<p>' ).html( ancillaryMessage );
			} )
		);

		// Events
		// (none)

		// Initialization
		// The following classes are used here:
		// * wbqc-report-status-violation
		// * wbqc-report-status-warning
		// * wbqc-report-status-suggestion
		this.$element
			.addClass( 'wbqc-report' )
			.addClass( 'wbqc-report-status-' + this.status );
		this.$headingLinks
			.append( this.$helpLink )
			.append( ' ' )
			.append( this.$discussLink );
		this.$heading.append( this.$headingLinks );
		this.$element.append( this.$heading );
		this.$element.append( this.$message );
		if ( this.$constraintClarification ) {
			this.$element.append( this.$constraintClarification );
		}
		this.$element.append( this.$ancillaryMessages );
	};

	/* Setup */

	OO.inheritClass( wb.quality.constraints.ui.ConstraintReportPanel, OO.ui.PanelLayout );

	/* Methods */

	// (none)

}( mediaWiki, wikibase ) );
