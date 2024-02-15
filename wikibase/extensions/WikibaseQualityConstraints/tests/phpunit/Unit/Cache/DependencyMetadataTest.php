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
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\DependencyMetadata
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class DependencyMetadataTest extends \MediaWikiUnitTestCase {

	public function testBlank() {
		$dm = DependencyMetadata::blank();

		$this->assertSame( [], $dm->getEntityIds() );
	}

	public function testOfEntityId() {
		$q42 = new ItemId( 'Q42' );
		$dm = DependencyMetadata::ofEntityId( $q42 );

		$this->assertSame( [ $q42 ], $dm->getEntityIds() );
	}

	public function testOfFutureTime() {
		$future = new TimeValue(
			'+2117-02-15T00:00:00Z',
			0,
			0,
			0,
			TimeValue::PRECISION_DAY,
			TimeValue::CALENDAR_GREGORIAN
		);
		$dm = DependencyMetadata::ofFutureTime( $future );

		$this->assertSame( $future, $dm->getFutureTime() );
	}

	public function testMerge_entityIds() {
		$q42 = new ItemId( 'Q42' );
		$p31 = new NumericPropertyId( 'P31' );
		$dm = DependencyMetadata::merge( [
			DependencyMetadata::blank(),
			DependencyMetadata::ofEntityId( $q42 ),
			DependencyMetadata::blank(),
			DependencyMetadata::ofEntityId( $p31 ),
		] );

		$expectedEntityIds = [ $q42, $p31 ];
		$actualEntityIds = $dm->getEntityIds();
		sort( $expectedEntityIds );
		sort( $actualEntityIds );
		$this->assertSame( $expectedEntityIds, $actualEntityIds );
	}

	public function testMerge_entityIds_deduplicate() {
		$q42 = new ItemId( 'Q42' );
		$p31 = new NumericPropertyId( 'P31' );
		$dm = DependencyMetadata::merge( [
			DependencyMetadata::ofEntityId( $q42 ),
			DependencyMetadata::blank(),
			DependencyMetadata::ofEntityId( $p31 ),
			DependencyMetadata::ofEntityId( $q42 ),
			DependencyMetadata::ofEntityId( $q42 ),
			DependencyMetadata::ofEntityId( $p31 ),
			DependencyMetadata::ofEntityId( $p31 ),
			DependencyMetadata::ofEntityId( $q42 ),
			DependencyMetadata::blank(),
			DependencyMetadata::ofEntityId( $p31 ),
			DependencyMetadata::ofEntityId( $q42 ),
			DependencyMetadata::ofEntityId( $p31 ),
			DependencyMetadata::ofEntityId( $p31 ),
			DependencyMetadata::ofEntityId( $p31 ),
			DependencyMetadata::ofEntityId( $q42 ),
		] );

		$expectedEntityIds = [ $q42, $p31 ];
		$actualEntityIds = $dm->getEntityIds();
		sort( $expectedEntityIds );
		sort( $actualEntityIds );
		$this->assertSame( $expectedEntityIds, $actualEntityIds );
	}

	public function testMerge_futureTime() {
		$nearFuture = new TimeValue(
			'+2027-02-15T00:00:00Z',
			0,
			0,
			0,
			TimeValue::PRECISION_DAY,
			TimeValue::CALENDAR_GREGORIAN
		);
		$intermediateFuture = new TimeValue(
			'+2117-02-15T00:00:00Z',
			0,
			0,
			0,
			TimeValue::PRECISION_DAY,
			TimeValue::CALENDAR_GREGORIAN
		);
		$farFuture = new TimeValue(
			'+3017-02-15T00:00:00Z',
			0,
			0,
			0,
			TimeValue::PRECISION_DAY,
			TimeValue::CALENDAR_GREGORIAN
		);

		$dm = DependencyMetadata::merge( [
			DependencyMetadata::blank(),
			DependencyMetadata::ofFutureTime( $intermediateFuture ),
			DependencyMetadata::ofFutureTime( $nearFuture ),
			DependencyMetadata::blank(),
			DependencyMetadata::ofFutureTime( $farFuture ),
		] );

		$this->assertSame( $nearFuture, $dm->getFutureTime() );
	}

	public function testMerge_differentFields() {
		$q42 = new ItemId( 'Q42' );
		$future = new TimeValue(
			'+2117-02-15T00:00:00Z',
			0,
			0,
			0,
			TimeValue::PRECISION_DAY,
			TimeValue::CALENDAR_GREGORIAN
		);

		$dm = DependencyMetadata::merge( [
			DependencyMetadata::ofEntityId( $q42 ),
			DependencyMetadata::ofFutureTime( $future ),
		] );

		$this->assertSame( [ $q42 ], $dm->getEntityIds() );
		$this->assertSame( $future, $dm->getFutureTime() );
	}

	public function testMerge_blank() {
		$cm = DependencyMetadata::merge( [
			DependencyMetadata::blank(),
			DependencyMetadata::blank(),
		] );

		$this->assertSame( [], $cm->getEntityIds() );
	}

	public function testMerge_invalid() {
		$this->expectException( ParameterElementTypeException::class );

		DependencyMetadata::merge( [
			[ new ItemId( 'Q42' ) ],
			Metadata::blank(),
			CachingMetadata::fresh(),
		] );
	}

}
