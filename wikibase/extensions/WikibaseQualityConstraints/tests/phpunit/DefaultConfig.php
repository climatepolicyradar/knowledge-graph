<?php

namespace WikibaseQuality\ConstraintReport\Tests;

use HashConfig;

/**
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
trait DefaultConfig {

	private static ?HashConfig $defaultConfig = null;

	public static function getDefaultConfig(): HashConfig {
		if ( self::$defaultConfig === null ) {
			self::$defaultConfig = new HashConfig();
			$extensionJsonFile = __DIR__ . '/../../extension.json';
			$extensionJsonText = file_get_contents( $extensionJsonFile );
			$extensionJson = json_decode( $extensionJsonText, /* assoc = */ true );
			foreach ( $extensionJson['config'] as $key => $value ) {
				self::$defaultConfig->set( $key, $value['value'] );
			}
			// reduce some limits to make tests run faster
			self::$defaultConfig->set( 'WBQualityConstraintsTypeCheckMaxEntities', 10 );
			// never query remote servers
			self::$defaultConfig->set( 'WBQualityConstraintsSparqlEndpoint', 'http://localhost:65536/' );
		}

		return clone self::$defaultConfig;
	}

}
