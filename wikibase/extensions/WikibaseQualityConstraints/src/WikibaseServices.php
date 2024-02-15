<?php

namespace WikibaseQuality\ConstraintReport;

use MediaWiki\MediaWikiServices;
use Wikibase\DataModel\Services\Lookup\EntityLookup;
use Wikibase\DataModel\Services\Lookup\PropertyDataTypeLookup;

/**
 * These services should really be registered by Wikibase,
 * but Wikibase doesn’t register them yet
 * and we need them to be registered as services for some tests,
 * so for now we register them on Wikibase’ behalf.
 *
 * @license GPL-2.0-or-later
 */
class WikibaseServices {

	public const ENTITY_LOOKUP = 'WBQC_EntityLookup';
	public const PROPERTY_DATA_TYPE_LOOKUP = 'WBQC_PropertyDataTypeLookup';
	public const ENTITY_LOOKUP_WITHOUT_CACHE = 'WBQC_EntityLookupWithoutCache';

	private static function getService( ?MediaWikiServices $services, $name ) {
		if ( $services === null ) {
			$services = MediaWikiServices::getInstance();
		}
		return $services->getService( $name );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return EntityLookup
	 */
	public static function getEntityLookup( MediaWikiServices $services = null ) {
		return self::getService( $services, self::ENTITY_LOOKUP );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return PropertyDataTypeLookup
	 */
	public static function getPropertyDataTypeLookup( MediaWikiServices $services = null ) {
		return self::getService( $services, self::PROPERTY_DATA_TYPE_LOOKUP );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return EntityLookup
	 */
	public static function getEntityLookupWithoutCache( MediaWikiServices $services = null ) {
		return self::getService( $services, self::ENTITY_LOOKUP_WITHOUT_CACHE );
	}

}
