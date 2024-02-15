<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Cache;

use DataValues\TimeValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachingMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\DependencyMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use Wikimedia\Assert\ParameterElementTypeException;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class MetadataTest extends \MediaWikiUnitTestCase {

	public function testBlank() {
		$m = Metadata::blank();

		$this->assertFalse( $m->getCachingMetadata()->isCached() );
		$this->assertSame( [], $m->getDependencyMetadata()->getEntityIds() );
	}

	public function testOfCachingMetadata() {
		$cm = CachingMetadata::ofMaximumAgeInSeconds( 42 );
		$m = Metadata::ofCachingMetadata( $cm );

		$this->assertSame( $cm, $m->getCachingMetadata() );
	}

	public function testOfDependencyMetadata() {
		$dm = DependencyMetadata::ofEntityId( new ItemId( 'Q42' ) );
		$m = Metadata::ofDependencyMetadata( $dm );

		$this->assertSame( $dm, $m->getDependencyMetadata() );
	}

	public function testMerge() {
		$q42 = new ItemId( 'Q42' );
		$p31 = new NumericPropertyId( 'P31' );
		$future = new TimeValue(
			'+2117-02-15T00:00:00Z',
			0,
			0,
			0,
			TimeValue::PRECISION_DAY,
			TimeValue::CALENDAR_GREGORIAN
		);
		$m = Metadata::merge( [
			Metadata::blank(),
			Metadata::ofCachingMetadata( CachingMetadata::fresh() ),
			Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 10 ) ),
			Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 42 ) ),
			Metadata::ofDependencyMetadata( DependencyMetadata::ofEntityId( $q42 ) ),
			Metadata::ofDependencyMetadata( DependencyMetadata::blank() ),
			Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 13 ) ),
			Metadata::ofDependencyMetadata( DependencyMetadata::ofEntityId( $p31 ) ),
			Metadata::ofDependencyMetadata( DependencyMetadata::ofFutureTime( $future ) ),
		] );

		$this->assertTrue( $m->getCachingMetadata()->isCached() );
		$this->assertSame( 42, $m->getCachingMetadata()->getMaximumAgeInSeconds() );
		$expectedEntityIds = [ $q42, $p31 ];
		$actualEntityIds = $m->getDependencyMetadata()->getEntityIds();
		sort( $expectedEntityIds );
		sort( $actualEntityIds );
		$this->assertSame( $expectedEntityIds, $actualEntityIds );
		$this->assertSame( $future, $m->getDependencyMetadata()->getFutureTime() );
	}

	public function testMerge_blank() {
		$m = Metadata::merge( [
			Metadata::blank(),
			Metadata::blank(),
		] );

		$this->assertFalse( $m->getCachingMetadata()->isCached() );
		$this->assertSame( [], $m->getDependencyMetadata()->getEntityIds() );
	}

	public function testMerge_invalid() {
		$this->expectException( ParameterElementTypeException::class );

		Metadata::merge( [
			10,
			[ new ItemId( 'Q42' ) ],
			CachingMetadata::fresh(),
			DependencyMetadata::blank(),
		] );
	}

	/**
	 * Make sure metadata objects can be compared with assertEquals(), some other tests rely on this
	 */
	public function testMerge_blankEquals() {
		$m = Metadata::merge( [ Metadata::blank() ] );

		$this->assertEquals( Metadata::blank(), $m );
	}

}
