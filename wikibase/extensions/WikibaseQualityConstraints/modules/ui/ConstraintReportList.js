( function ( OO, wb ) {
	'use strict';

	/**
	 * A panel for a list of constraint report results grouped by status.
	 *
	 * Usually, you want to initialize the list directly from a list of constraint reports.
	 * Use the static method {@link wikibase.quality.constraints.ui.ConstraintReportList.fromPanels} for that:
	 *
	 *     @example
	 *     var report = new wikibase.quality.constraints.ui.ConstraintReportList.static.fromPanels(
	 *         [
	 *             new wikibase.quality.constraints.ui.ConstraintReportPanel( {
	 *                 status: 'compliance',
	 *                 constraint: {
	 *                     type: 'Q1',
	 *                     typeLabel: 'my constraint',
	 *                     link: 'http://example.com/my-constraint',
	 *                     discussLink: 'https://wikidata.org/Property_talk:P2'
	 *                 },
	 *                 message: 'everything okay'
	 *             } ),
	 *             new wikibase.quality.constraints.ui.ConstraintReportPanel( {
	 *                 status: 'violation',
	 *                 constraint: {
	 *                     type: 'Q2',
	 *                     typeLabel: 'my other constraint',
	 *                     link: 'http://example.com/my-other-constraint',
	 *                     discussLink: 'http://example.com/Property_talk:P2'
	 *                 },
	 *                 message: 'doing it wrong'
	 *             } )
	 *         ],
	 *         {
	 *             statuses: [
	 *                 {
	 *                     status: 'violation',
	 *                     label: 'Violations'
	 *                 },
	 *                 {
	 *                     status: 'bad-parameters',
	 *                     label: 'Bad parameters',
	 *                     collapsed: true
	 *                 }
	 *             ],
	 *             expanded: false,
	 *             framed: true
	 *         }
	 *     );
	 *     $( 'body' ).append( report.$element );
	 *
	 * You can also explicitly provide the groups to the constructor:
	 *
	 *     @example
	 *     var report = new wikibase.quality.constraints.ui.ConstraintReportList( {
	 *         items: [
	 *             new wikibase.quality.constraints.ui.ConstraintReportGroup( {
	 *                 items: [
	 *                     new wikibase.quality.constraints.ui.ConstraintReportPanel( {
	 *                         status: 'compliance',
	 *                         constraint: {
	 *                             type: 'Q1',
	 *                             typeLabel: 'my constraint',
	 *                             link: 'http://example.com/my-constraint',
	 *                             discussLink: 'http://example.com/Property_talk:P2'
	 *                         },
	 *                         message: 'everything okay'
	 *                     } )
	 *                 ],
	 *                 label: 'Violations'
	 *             } ),
	 *             new wikibase.quality.constraints.ui.ConstraintReportGroup( {
	 *                 items: [
	 *                     new wikibase.quality.constraints.ui.ConstraintReportPanel( {
	 *                         status: 'violation',
	 *                         constraint: {
	 *                             type: 'Q2',
	 *                             typeLabel: 'my other constraint',
	 *                             link: 'http://example.com/my-other-constraint',
	 *                             discussLink: 'http://example.com/Property_talk:P2'
	 *                         },
	 *                         message: 'doing it wrong'
	 *                     } )
	 *                 ],
	 *                 label: 'Bad parameters',
	 *                 collapsed: true
	 *             } )
	 *         ],
	 *         expanded: false,
	 *         framed: true
	 *     } );
	 *     $( 'body' ).append( report.$element );
	 *
	 * @class
	 * @extends OO.ui.StackLayout
	 *
	 * @constructor
	 * @param {Object} config Configuration options
	 * @cfg {wikibase.quality.constraints.ui.ConstraintReportGroup[]} items The constraint report groups.
	 */
	wb.quality.constraints.ui.ConstraintReportList = function WBQCConstraintReportList( config ) {

		// Configuration initialization
		config.continuous = true;
		config.classes = OO.simpleArrayUnion(
			config.classes || [],
			[ 'wbqc-reports-all' ]
		);

		// Parent constructor
		wb.quality.constraints.ui.ConstraintReportList.parent.call( this, config );

		// Mixin constructors
		// (none)
	};

	/* Setup */

	OO.inheritClass( wb.quality.constraints.ui.ConstraintReportList, OO.ui.StackLayout );

	/* Methods */

	/**
	 * Sort a flat list of constraint reports into groups by status.
	 *
	 * @param {wikibase.quality.constraints.ui.ConstraintReportPanel[]} panels The individual constraint report panels.
	 * @param {Object} config Configuration options (all options except `statuses` are passed to the `ConstraintReportList` constructor).
	 * @param {Object[]} config.statuses The configuration for each group.
	 * The `status` member selects constraint report panels from `panels` with this status;
	 * all other options are passed to the `ConstraintReportGroup` constructor.
	 * The order of status objects in the array determines the order of groups in the list.
	 * @return {wikibase.quality.constraints.ui.ConstraintReportList}
	 * @static
	 */
	wb.quality.constraints.ui.ConstraintReportList.static.fromPanels = function ( panels, config ) {
		var panelsByStatus = {},
			items = [];

		panels.forEach( function ( panel ) {
			panelsByStatus[ panel.status ] = panelsByStatus[ panel.status ] || [];
			panelsByStatus[ panel.status ].push( panel );
		} );

		config.statuses.forEach( function ( statusConfig ) {
			var status = statusConfig.status,
				statusPanels = panelsByStatus[ status ];

			if ( statusPanels === undefined ) {
				return;
			}

			delete statusConfig.status; // remainder of config is for ConstraintReportGroup
			statusConfig.items = statusPanels;
			statusConfig.classes = OO.simpleArrayUnion(
				statusConfig.classes || [],
				[ 'wbqc-reports-status-' + status ]
			);
			statusConfig.status = status;
			items.push( new wb.quality.constraints.ui.ConstraintReportGroup( statusConfig ) );
		} );

		delete config.statuses; // remainder of config is for ConstraintReportList
		config.items = items;
		return new wb.quality.constraints.ui.ConstraintReportList( config );
	};

}( OO, wikibase ) );
