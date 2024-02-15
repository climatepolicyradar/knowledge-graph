<?php

namespace WikibaseQuality\ConstraintReport;

use MediaWiki\MediaWikiServices;
use Wikibase\DataModel\Services\Lookup\ExceptionIgnoringEntityLookup;
use Wikibase\Repo\Store\Store;
use Wikibase\Repo\WikibaseRepo;

return [
	WikibaseServices::ENTITY_LOOKUP => static function ( MediaWikiServices $services ) {
		return new ExceptionIgnoringEntityLookup(
			WikibaseRepo::getEntityLookup( $services )
		);
	},

	WikibaseServices::ENTITY_LOOKUP_WITHOUT_CACHE => static function ( MediaWikiServices $services ) {
		return new ExceptionIgnoringEntityLookup(
			WikibaseRepo::getStore( $services )
				->getEntityLookup( Store::LOOKUP_CACHING_RETRIEVE_ONLY )
		);
	},

	WikibaseServices::PROPERTY_DATA_TYPE_LOOKUP => static function ( MediaWikiServices $services ) {
		return WikibaseRepo::getPropertyDataTypeLookup( $services );
	},
];
