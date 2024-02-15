<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Cache;

use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedBool;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachingMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedBool
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CachedBoolTest extends \MediaWikiUnitTestCase {

	public function testGetBool() {
		$bool = true;
		$cm = Metadata::blank();

		$cb = new CachedBool( $bool, $cm );

		$this->assertSame( $bool, $cb->getBool() );
	}

	public function testGetMetadata() {
		$bool = false;
		$m = Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 42 ) );

		$cb = new CachedBool( $bool, $m );

		$this->assertSame( $m, $cb->getMetadata() );
	}

}
