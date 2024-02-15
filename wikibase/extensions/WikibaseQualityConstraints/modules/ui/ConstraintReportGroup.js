( function ( OO, wb ) {
	'use strict';

	/**
	 * A panel for a group of constraint report results.
	 *
	 *     @example
	 *     var group = new wikibase.quality.constraints.ui.ConstraintReportGroup( {
	 *         items: [
	 *             new wikibase.quality.constraints.ui.ConstraintReportPanel( {
	 *                 status: 'compliance',
	 *                 constraint: {
	 *                     type: 'Q1',
	 *                     typeLabel: 'my constraint',
	 *                     link: 'http://example.com/my-constraint'
	 *                 },
	 *                 message: 'everything okay'
	 *             } )
	 *         ],
	 *         label: 'reports that are fine',
	 *         collapsed: true,
	 *         expanded: false,
	 *         framed: true
	 *     } );
	 *     $( 'body' ).append( group.$element );
	 *
	 * @class
	 * @extends OO.ui.StackLayout
	 * eslint-disable-next-line jsdoc/check-tag-names T268378
	 * @mixins OO.ui.mixin.LabelElement
	 *
	 * @constructor
	 * @param {Object} config Configuration options
	 * @cfg {wikibase.quality.constraints.ui.ConstraintReportPanel[]} items The individual constraint report results.
	 * @cfg {string} [subheading] The subheading of the group. Unlike the label, it is part of the collapsed content if `collapsible` is `true`.
	 * @cfg {boolean} [collapsible=false] Whether the group can be collapsed or not. If `true`, a `label` is recommended.
	 * @cfg {boolean} [collapsed=false] Whether the group is collapsed by default or not. (If `true`, implies `collapsible=true` as well.)
	 * @cfg {string} [status] The status of all reports in this group. Unused by this class, but assigned by `ConstraintReportList.fromPanels`.
	 */
	wb.quality.constraints.ui.ConstraintReportGroup = function WBQCConstraintReportGroup( config ) {

		var collapsible = config.collapsible || config.collapsed || false,
			collapsed = config.collapsed || false,
			subheading = config.subheading || null,
			status = config.status || null,
			$subheadingContainer;

		// Configuration initialization
		config.continuous = true;
		config.classes = OO.simpleArrayUnion(
			config.classes || [],
			[ 'wbqc-reports' ]
		);
		if ( typeof config.label === 'string' ) {
			config.label = $( '<strong>' ).text( config.label );
		}
		if ( collapsible && config.label === undefined ) {
			// blank, but nonempty label so the collapse toggler isn't hidden behind the items
			config.label = ' ';
		}

		// Parent constructor
		wb.quality.constraints.ui.ConstraintReportGroup.parent.call( this, config );

		// Mixin constructors
		OO.ui.mixin.LabelElement.call( this, config );

		// Properties
		this.collapsible = collapsible;
		this.collapsed = collapsed;
		this.status = status;

		// Initialization
		if ( collapsible ) {
			this.$element.makeCollapsible( { collapsed: collapsed } );
		}
		this.$element.prepend( this.$label );
		if ( subheading !== null ) {
			$subheadingContainer = collapsible ?
				this.$element.find( '.mw-collapsible-content' ) :
				this.$element;
			$subheadingContainer.prepend(
				$( '<p>' ).append( $( '<small>' ).text( subheading ) )
			);
		}
	};

	/* Setup */

	OO.inheritClass( wb.quality.constraints.ui.ConstraintReportGroup, OO.ui.StackLayout );
	OO.mixinClass( wb.quality.constraints.ui.ConstraintReportGroup, OO.ui.mixin.LabelElement );

	/* Methods */

	// (none)

}( OO, wikibase ) );
