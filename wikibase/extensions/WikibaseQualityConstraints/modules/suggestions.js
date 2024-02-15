( function ( mw, wb, $ ) {
	'use strict';
	var config = require( './config.json' ),
		makeHandler = require( './suggestions/EntitySelectorHookHandlerFactory.js' );

	function isQualifierContext( element ) {
		var $statementview, statementview;

		try {
			$statementview = element.closest( ':wikibase-statementview' );
			statementview = $statementview.data( 'statementview' );

			if ( !statementview ) {
				return false;
			}

			return element.closest( statementview.$qualifiers ).length > 0;

		} catch ( e ) {
			return false;
		}
	}

	function getMainSnakPropertyId( element ) {
		var snakview, snakPropertyId;

		try {
			snakview = element.closest( '.wikibase-statementlistview' )
				.find( '.wikibase-snakview.wb-edit' ).data( 'snakview' );
			snakPropertyId = snakview ? snakview.propertyId() : null;
		} catch ( e ) {
			return null;
		}

		return snakPropertyId;
	}

	mw.hook( 'wikibase.entityselector.search' ).add( makeHandler( $, mw, config, isQualifierContext, getMainSnakPropertyId ) );

}( mediaWiki, wikibase, jQuery ) );
