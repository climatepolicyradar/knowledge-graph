<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Api;

use HashBagOStuff;
use WANObjectCache;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use WikibaseQuality\ConstraintReport\Api\ResultsCache;

/**
 * @covers WikibaseQuality\ConstraintReport\Api\ResultsCache
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ResultsCacheTest extends \MediaWikiUnitTestCase {

	public function testMakeKey() {
		$resultsCache = new ResultsCache( WANObjectCache::newEmpty(), 'v2' );

		$this->assertSame(
			'local:WikibaseQualityConstraints:checkConstraints:v2:Q5',
			$resultsCache->makeKey( new ItemId( 'Q5' ) )
		);
		$this->assertSame(
			'local:WikibaseQualityConstraints:checkConstraints:v2:P31',
			$resultsCache->makeKey( new NumericPropertyId( 'P31' ) )
		);
	}

	public function testMakeKey_differentFormatVersion() {
		$resultsCache = new ResultsCache( WANObjectCache::newEmpty(), 'v3' );

		$key = $resultsCache->makeKey( new ItemId( 'Q5' ) );

		$this->assertSame( 'local:WikibaseQualityConstraints:checkConstraints:v3:Q5', $key );
	}

	public function testGet() {
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$q5 = new ItemId( 'Q5' );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$expectedValue = 'garbage data, should not matter';
		$cache->set( $resultsCache->makeKey( $q5 ), $expectedValue );

		$actualValue = $resultsCache->get( $q5 );

		$this->assertSame( $expectedValue, $actualValue );
	}

	public function testSet() {
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$q5 = new ItemId( 'Q5' );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$expectedValue = 'garbage data, should not matter';

		$resultsCache->set( $q5, $expectedValue );

		$actualValue = $cache->get( $resultsCache->makeKey( $q5 ) );
		$this->assertSame( $expectedValue, $actualValue );
	}

	public function testGetSet() {
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$q5 = new ItemId( 'Q5' );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$expectedValue = 'garbage data, should not matter';

		$resultsCache->set( $q5, $expectedValue );
		$actualValue = $resultsCache->get( $q5 );

		$this->assertSame( $expectedValue, $actualValue );
	}

	public function testDelete() {
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$q5 = new ItemId( 'Q5' );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$expectedValue = 'garbage data, should not matter';
		$cache->set( $resultsCache->makeKey( $q5 ), $expectedValue );

		$resultsCache->delete( $q5 );

		$this->assertFalse( $cache->get( $resultsCache->makeKey( $q5 ) ) );
	}

}
