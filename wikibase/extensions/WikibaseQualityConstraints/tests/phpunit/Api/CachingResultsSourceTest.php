<?php

namespace WikibaseQuality\ConstraintReport\Tests\Api;

use DataValues\TimeValue;
use HashBagOStuff;
use HashConfig;
use NullStatsdDataFactory;
use Psr\Log\NullLogger;
use WANObjectCache;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\ItemIdParser;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\Lib\Store\LookupConstants;
use Wikibase\Lib\Store\Sql\WikiPageEntityMetaDataAccessor;
use WikibaseQuality\ConstraintReport\Api\CachingResultsSource;
use WikibaseQuality\ConstraintReport\Api\ResultsCache;
use WikibaseQuality\ConstraintReport\Api\ResultsSource;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckResults;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachingMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\DependencyMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\EntityContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\LoggingHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\NullResult;

/**
 * @covers WikibaseQuality\ConstraintReport\Api\CachingResultsSource
 *
 * @license GPL-2.0-or-later
 */
class CachingResultsSourceTest extends \PHPUnit\Framework\TestCase {

	/**
	 * @param string $entityId entity ID serialization
	 */
	private function getCheckResult( $entityId, $status = CheckResult::STATUS_VIOLATION ) {
		return new CheckResult(
			new MainSnakContextCursor(
				$entityId,
				'P1',
				$entityId . '$00000000-0000-0000-0000-000000000000',
				'0000000000000000000000000000000000000000'
			),
			new Constraint(
				'P1$00000000-0000-0000-0000-000000000000',
				new NumericPropertyId( 'P1' ),
				'Q12345',
				[]
			),
			$status,
			new ViolationMessage( 'wbqc-violation-message-single-value' )
		);
	}

	/**
	 * @return CheckResultSerializer
	 */
	private function getCheckResultSerializer() {
		$mock = $this->createMock( CheckResultSerializer::class );
		$mock->method( 'serialize' )
			->willReturnCallback( function ( CheckResult $checkResult ) {
				$entityId = $checkResult->getContextCursor()->getEntityId();
				if ( $checkResult instanceof NullResult ) {
					return [ 'NullResult for ' . $entityId ];
				} else {
					return [ 'CheckResult for ' . $entityId ];
				}
			} );
		return $mock;
	}

	/**
	 * @return CheckResultDeserializer
	 */
	private function getCheckResultDeserializer() {
		$mock = $this->createMock( CheckResultDeserializer::class );
		$mock->method( 'deserialize' )
			->willReturnCallback( function ( $serialization ) {
				if ( strpos( $serialization[0], 'NullResult for ' ) === 0 ) {
					$id = str_replace( 'NullResult for ', '', $serialization[0] );
					return new NullResult( new EntityContextCursor( $id ) );
				} else {
					$id = str_replace( 'CheckResult for ', '', $serialization[0] );
					return $this->getCheckResult( $id );
				}
			} );
		return $mock;
	}

	/**
	 * @return LoggingHelper
	 */
	private function getLoggingHelper() {
		return new LoggingHelper(
			new NullStatsdDataFactory(),
			new NullLogger(),
			new HashConfig( [
				'WBQualityConstraintsCheckDurationInfoSeconds' => 5.0,
				'WBQualityConstraintsCheckDurationWarningSeconds' => 10.0,
				'WBQualityConstraintsCheckOnEntityDurationInfoSeconds' => 15.0,
				'WBQualityConstraintsCheckOnEntityDurationWarningSeconds' => 55.0,
			] )
		);
	}

	/**
	 * @param ResultsSource $resultsSource
	 * @param ResultsCache $resultsCache
	 * @param WikiPageEntityMetaDataAccessor $metaDataAccessor
	 * @param string[] $possiblyStaleConstraintTypes
	 * @return CachingResultsSource
	 */
	private function getCachingResultsSource(
		ResultsSource $resultsSource,
		ResultsCache $resultsCache,
		WikiPageEntityMetaDataAccessor $metaDataAccessor,
		array $possiblyStaleConstraintTypes = []
	) {
		return new CachingResultsSource(
			$resultsSource,
			$resultsCache,
			$this->getCheckResultSerializer(),
			$this->getCheckResultDeserializer(),
			$metaDataAccessor,
			new ItemIdParser(),
			86400,
			$possiblyStaleConstraintTypes,
			10000,
			$this->getLoggingHelper()
		);
	}

	private function getCachingResultsSourceMock(
		ResultsSource $resultsSource,
		ResultsCache $resultsCache,
		WikiPageEntityMetaDataAccessor $metaDataAccessor
	) {
		return $this->getMockBuilder( CachingResultsSource::class )
			->setConstructorArgs( [
				$resultsSource,
				$resultsCache,
				$this->getCheckResultSerializer(),
				$this->getCheckResultDeserializer(),
				$metaDataAccessor,
				new ItemIdParser(),
				86400,
				[],
				10000,
				$this->getLoggingHelper(),
			] )
			->onlyMethods( [ 'getStoredResults', 'getAndStoreResults' ] )
			->getMock();
	}

	public function testGetAndStoreResults_SameResults() {
		$expectedResults = new CachedCheckResults(
			[ $this->getCheckResult( 'Q100' ) ],
			Metadata::blank()
		);
		$q100 = new ItemId( 'Q100' );
		$statuses = [ 'garbage status', 'other status' ];
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [], null, $statuses )
			->willReturn( $expectedResults );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [] );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			new ResultsCache( WANObjectCache::newEmpty(), 'v2' ),
			$metaDataAccessor
		);

		$results = $cachingResultsSource->getAndStoreResults( [ $q100 ], [], null, $statuses );

		$this->assertSame( $expectedResults, $results );
	}

	public function testGetAndStoreResults_DontCacheClaimIds() {
		$expectedResults = new CachedCheckResults(
			[ $this->getCheckResult( 'Q100' ) ],
			Metadata::blank()
		);
		$q100 = new ItemId( 'Q100' );
		$statuses = [ 'garbage status', 'other status' ];
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [ 'fake' ], null, $statuses )
			->willReturn( $expectedResults );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->expects( $this->never() )->method( 'loadLatestRevisionIds' );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			new ResultsCache( $cache, 'v2' ),
			$metaDataAccessor
		);

		$cachingResultsSource->getAndStoreResults( [ $q100 ], [ 'fake' ], null, $statuses );

		$this->assertNull( $cachingResultsSource->getStoredResults( $q100 ) );
	}

	public function testGetAndStoreResults_DontCacheWithConstraintIds() {
		$expectedResults = new CachedCheckResults(
			[ $this->getCheckResult( 'Q100' ) ],
			Metadata::blank()
		);
		$q100 = new ItemId( 'Q100' );
		$statuses = [ 'garbage status', 'other status' ];
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [], [ 'fake' ], $statuses )
			->willReturn( $expectedResults );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->expects( $this->never() )->method( 'loadLatestRevisionIds' );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			new ResultsCache( $cache, 'v2' ),
			$metaDataAccessor
		);

		$cachingResultsSource->getAndStoreResults( [ $q100 ], [], [ 'fake' ], $statuses );

		$this->assertNull( $cachingResultsSource->getStoredResults( $q100 ) );
	}

	public function testGetAndStoreResults_StoreResults() {
		$expectedResults = new CachedCheckResults(
			[ $this->getCheckResult( 'Q100' ) ],
			Metadata::blank()
		);
		$q100 = new ItemId( 'Q100' );
		$statuses = CachingResultsSource::CACHED_STATUSES;
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [], null, $statuses )
			->willReturn( $expectedResults );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [] );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			$resultsCache,
			$metaDataAccessor
		);

		$cachingResultsSource->getAndStoreResults( [ $q100 ], [], null, $statuses );
		$cachedResults = $resultsCache->get( $q100 );

		$this->assertNotFalse( $cachedResults );
		$this->assertArrayHasKey( 'results', $cachedResults );
		$this->assertCount( 1, $cachedResults['results'] );
		$this->assertSame( [ 'CheckResult for Q100' ], $cachedResults['results'][0] );
	}

	public function testGetAndStoreResults_StoreLatestRevisionIds() {
		$q100 = new ItemId( 'Q100' );
		$q101 = new ItemId( 'Q101' );
		$p102 = new NumericPropertyId( 'P102' );
		$expectedResults = new CachedCheckResults(
			[ $this->getCheckResult( 'Q100' ) ],
			Metadata::ofDependencyMetadata( DependencyMetadata::merge( [
				DependencyMetadata::ofEntityId( $q100 ),
				DependencyMetadata::ofEntityId( $q101 ),
				DependencyMetadata::ofEntityId( $p102 ),
			] ) )
		);
		$revisionIds = [
			$q100->getSerialization() => 12345,
			$q101->getSerialization() => 1337,
			$p102->getSerialization() => 42,
		];
		$statuses = CachingResultsSource::CACHED_STATUSES;
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [], null, $statuses )
			->willReturn( $expectedResults );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->expects( $this->once() )
			->method( 'loadLatestRevisionIds' )
			->with(
				[ $q100, $q101, $p102 ],
				LookupConstants::LATEST_FROM_REPLICA
			)
			->willReturn( $revisionIds );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			$resultsCache,
			$metaDataAccessor
		);

		$cachingResultsSource->getAndStoreResults( [ $q100 ], [], null, $statuses );
		$cachedResults = $resultsCache->get( $q100 );

		$this->assertNotFalse( $cachedResults );
		$this->assertArrayHasKey( 'latestRevisionIds', $cachedResults );
		$this->assertSame( $revisionIds, $cachedResults['latestRevisionIds'] );
	}

	public function testGetAndStoreResults_StoreWithExtraStatuses() {
		$expectedResults = new CachedCheckResults(
			[
				$this->getCheckResult( 'Q100' ),
				$this->getCheckResult( 'Q100', CheckResult::STATUS_TODO ),
			],
			Metadata::blank()
		);
		$q100 = new ItemId( 'Q100' );
		$statuses = array_merge(
			CachingResultsSource::CACHED_STATUSES,
			[ CheckResult::STATUS_TODO ]
		);
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [], null, $statuses )
			->willReturn( $expectedResults );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [] );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			$resultsCache,
			$metaDataAccessor
		);

		$cachingResultsSource->getAndStoreResults( [ $q100 ], [], null, $statuses );
		$cachedResults = $resultsCache->get( $q100 );

		$this->assertNotFalse( $cachedResults );
		$this->assertArrayHasKey( 'results', $cachedResults );
		$this->assertCount( 1, $cachedResults['results'] );
		$this->assertSame( [ 'CheckResult for Q100' ], $cachedResults['results'][0] );
	}

	public function testGetAndStoreResults_DontStoreWithMissingStatuses() {
		$expectedResults = new CachedCheckResults(
			[ $this->getCheckResult( 'Q100' ) ],
			Metadata::blank()
		);
		$q100 = new ItemId( 'Q100' );
		$statuses = [ CheckResult::STATUS_VIOLATION ];
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [], null, $statuses )
			->willReturn( $expectedResults );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->expects( $this->never() )->method( 'loadLatestRevisionIds' );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			new ResultsCache( $cache, 'v2' ),
			$metaDataAccessor
		);

		$cachingResultsSource->getAndStoreResults( [ $q100 ], [], null, $statuses );

		$this->assertNull( $cachingResultsSource->getStoredResults( $q100 ) );
	}

	public function testGetAndStoreResults_WithoutFutureTime() {
		$expectedResults = new CachedCheckResults(
			[ $this->getCheckResult( 'Q100' ) ],
			Metadata::ofDependencyMetadata( DependencyMetadata::blank() )
		);
		$q100 = new ItemId( 'Q100' );
		$statuses = CachingResultsSource::CACHED_STATUSES;
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [], null, $statuses )
			->willReturn( $expectedResults );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [] );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			$resultsCache,
			$metaDataAccessor
		);

		$cachingResultsSource->getAndStoreResults( [ $q100 ], [], null, $statuses );
		$cachedResults = $resultsCache->get( $q100 );

		$this->assertNotFalse( $cachedResults );
		$this->assertArrayNotHasKey( 'futureTime', $cachedResults );
	}

	public function testGetAndStoreResults_WithFutureTime() {
		$timeValue = new TimeValue(
			'+2012-10-29T00:00:00Z',
			0,
			0,
			0,
			TimeValue::PRECISION_DAY,
			TimeValue::CALENDAR_GREGORIAN
		);
		$expectedResults = new CachedCheckResults(
			[ $this->getCheckResult( 'Q100' ) ],
			Metadata::ofDependencyMetadata( DependencyMetadata::ofFutureTime( $timeValue ) )
		);
		$q100 = new ItemId( 'Q100' );
		$statuses = CachingResultsSource::CACHED_STATUSES;
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [], null, $statuses )
			->willReturn( $expectedResults );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [] );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			$resultsCache,
			$metaDataAccessor
		);

		$cachingResultsSource->getAndStoreResults( [ $q100 ], [], null, $statuses );
		$cachedResults = $resultsCache->get( $q100 );

		$this->assertNotFalse( $cachedResults );
		$this->assertArrayHasKey( 'futureTime', $cachedResults );
		$this->assertSame( $timeValue->getArrayValue(), $cachedResults['futureTime'] );
	}

	public function testGetAndStoreResults_DontStoreWithoutRevisionInformation() {
		$q100 = new ItemId( 'Q100' );
		$expectedResults = new CachedCheckResults(
			[ $this->getCheckResult( 'Q100' ) ],
			Metadata::ofDependencyMetadata( DependencyMetadata::ofEntityId( $q100 ) )
		);
		$statuses = CachingResultsSource::CACHED_STATUSES;
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [], null, $statuses )
			->willReturn( $expectedResults );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->expects( $this->once() )
			->method( 'loadLatestRevisionIds' )
			->with(
				[ $q100 ],
				LookupConstants::LATEST_FROM_REPLICA
			)
			->willReturn( [ 'Q100' => false ] );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			new ResultsCache( $cache, 'v2' ),
			$metaDataAccessor
		);

		$cachingResultsSource->getAndStoreResults( [ $q100 ], [], null, $statuses );

		$this->assertNull( $cachingResultsSource->getStoredResults( $q100 ) );
	}

	public function testGetAndStoreResults_NullResult() {
		$q100 = new ItemId( 'Q100' );
		$expectedResults = new CachedCheckResults(
			[ new NullResult( new EntityContextCursor( 'Q100' ) ) ],
			Metadata::ofDependencyMetadata( DependencyMetadata::ofEntityId( $q100 ) )
		);
		$statuses = CachingResultsSource::CACHED_STATUSES;
		$resultsSource = $this->createMock( ResultsSource::class );
		$resultsSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ $q100 ], [], null, $statuses )
			->willReturn( $expectedResults );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [] );
		$cachingResultsSource = $this->getCachingResultsSource(
			$resultsSource,
			$resultsCache,
			$metaDataAccessor
		);

		$cachingResultsSource->getAndStoreResults( [ $q100 ], [], null, $statuses );
		$cachedResults = $resultsCache->get( $q100 );

		$this->assertNotFalse( $cachedResults );
		$this->assertArrayHasKey( 'results', $cachedResults );
		$this->assertCount( 1, $cachedResults['results'] );
		$this->assertSame( [ 'NullResult for Q100' ], $cachedResults['results'][0] );
	}

	public function testGetStoredResults_CacheMiss() {
		$cachingResultsSource = $this->getCachingResultsSource(
			$this->createMock( ResultsSource::class ),
			new ResultsCache( WANObjectCache::newEmpty(), 'v2' ),
			$this->createMock( WikiPageEntityMetaDataAccessor::class )
		);

		$response = $cachingResultsSource->getStoredResults( new ItemId( 'Q1' ) );

		$this->assertNull( $response );
	}

	public function testGetStoredResults_Outdated() {
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [
				'Q5' => 100,
				'Q10' => 101,
			] );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$cachingResultsSource = $this->getCachingResultsSource(
			$this->createMock( ResultsSource::class ),
			$resultsCache,
			$metaDataAccessor
		);
		$q5 = new ItemId( 'Q5' );
		$value = [
			'results' => [ [ 'CheckResult for Q5' ] ],
			'latestRevisionIds' => [
				'Q5' => 100,
				'Q10' => 99,
			],
		];
		$resultsCache->set( $q5, $value );

		$response = $cachingResultsSource->getStoredResults( $q5 );

		$this->assertNull( $response );
	}

	public function testGetStoredResults_Fresh() {
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [
				'Q5' => 100,
				'Q10' => 99,
			] );

		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$now = 9001;
		$mockTime = $now - 1337;
		$cache->setMockTime( $mockTime );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$cachingResultsSource = $this->getCachingResultsSource(
			$this->createMock( ResultsSource::class ),
			$resultsCache,
			$metaDataAccessor
		);
		$cachingResultsSource->setMicrotimeFunction( function () use ( $now ) {
			return $now;
		} );

		$q5 = new ItemId( 'Q5' );
		$value = [
			'results' => [ [ 'CheckResult for Q5' ] ],
			'latestRevisionIds' => [
				'Q5' => 100,
				'Q10' => 99,
			],
		];
		$resultsCache->set( $q5, $value );

		$response = $cachingResultsSource->getStoredResults( $q5 );

		$this->assertNotNull( $response );
		$this->assertEquals( [ $this->getCheckResult( 'Q5' ) ], $response->getArray() );
		$cachingMetadata = $response->getMetadata()->getCachingMetadata();
		$this->assertTrue( $cachingMetadata->isCached() );
		$this->assertSame( 1337, $cachingMetadata->getMaximumAgeInSeconds() );
	}

	public function testGetStoredResults_WithoutRevisionInformation() {
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor
			->method( 'loadLatestRevisionIds' )
			->willReturn( [
				'Q5' => 100,
				'Q10' => false,
			] );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$cachingResultsSource = $this->getCachingResultsSource(
			$this->createMock( ResultsSource::class ),
			$resultsCache,
			$metaDataAccessor
		);
		$q5 = new ItemId( 'Q5' );
		$value = [
			'results' => [ [ 'CheckResult for Q5' ] ],
			'latestRevisionIds' => [
				'Q5' => 100,
				'Q10' => 99,
			],
		];
		$resultsCache->set( $q5, $value );

		$response = $cachingResultsSource->getStoredResults( $q5 );

		$this->assertNull( $response );
	}

	public function testGetStoredResults_UpdateCachingMetadata() {
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [
				'Q5' => 100,
				'Q10' => 99,
			] );

		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$now = 9001;
		$mockTime = $now - 1337;
		$cache->setMockTime( $mockTime );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$cachingResultsSource = $this->getCachingResultsSource(
			$this->createMock( ResultsSource::class ),
			$resultsCache,
			$metaDataAccessor,
			[ $this->getCheckResult( 'Q5' )->getConstraint()->getConstraintTypeItemId() ]
		);
		$cachingResultsSource->setMicrotimeFunction( function () use ( $now ) {
			return $now;
		} );

		$q5 = new ItemId( 'Q5' );
		$value = [
			'results' => [ [ 'CheckResult for Q5' ] ],
			'latestRevisionIds' => [
				'Q5' => 100,
				'Q10' => 99,
			],
		];
		$resultsCache->set( $q5, $value );

		$response = $cachingResultsSource->getStoredResults( $q5 );

		$this->assertNotNull( $response );
		$overallCachingMetadata = $response->getMetadata()->getCachingMetadata();
		$this->assertTrue( $overallCachingMetadata->isCached() );
		$this->assertSame( 1337, $overallCachingMetadata->getMaximumAgeInSeconds() );
		$resultCachingMetadata = $response->getArray()[0]->getMetadata()->getCachingMetadata();
		$this->assertTrue( $resultCachingMetadata->isCached() );
		$this->assertSame( 1337, $resultCachingMetadata->getMaximumAgeInSeconds() );
	}

	public function testGetStoredResults_NullResult() {
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [
				'Q5' => 100,
			] );
		$cache = new WANObjectCache( [ 'cache' => new HashBagOStuff() ] );
		$resultsCache = new ResultsCache( $cache, 'v2' );
		$cachingResultsSource = $this->getCachingResultsSource(
			$this->createMock( ResultsSource::class ),
			$resultsCache,
			$metaDataAccessor
		);
		$q5 = new ItemId( 'Q5' );
		$value = [
			'results' => [ [ 'NullResult for Q5' ] ],
			'latestRevisionIds' => [
				'Q5' => 100,
			],
		];
		$resultsCache->set( $q5, $value );

		$response = $cachingResultsSource->getStoredResults( $q5 );
		$results = $response->getArray();

		$this->assertCount( 1, $results );
		$expected = new NullResult( new EntityContextCursor( 'Q5' ) );
		$this->assertEquals( $expected, $results[0] );
	}

	public function testGetResults_EmptyCache() {
		$entityIds = [ new ItemId( 'Q5' ), new ItemId( 'Q10' ) ];
		$statuses = [ 'garbage status', 'other status' ];

		$cachingResultsSource = $this->getCachingResultsSourceMock(
			$this->createMock( ResultsSource::class ),
			new ResultsCache( WANObjectCache::newEmpty(), 'v2' ),
			$this->createMock( WikiPageEntityMetaDataAccessor::class )
		);
		$cachingResultsSource->method( 'getStoredResults' )->willReturn( null );
		$cachingResultsSource->method( 'getAndStoreResults' )
			->with( $entityIds, [], null, $statuses )
			->willReturnCallback( function ( $entityIds, $claimIds, $constraintIds ) {
				$results = [];
				foreach ( $entityIds as $entityId ) {
					$serialization = $entityId->getSerialization();
					$results[] = $this->getCheckResult( $serialization );
				}
				return new CachedCheckResults( $results, Metadata::blank() );
			} );
		/** @var CachingResultsSource $cachingResultsSource */

		$results = $cachingResultsSource->getResults(
			$entityIds,
			[],
			null,
			$statuses
		);

		$expected = [ $this->getCheckResult( 'Q5' ), $this->getCheckResult( 'Q10' ) ];
		$actual = $results->getArray();
		sort( $expected );
		sort( $actual );
		$this->assertEquals( $expected, $actual );
		$this->assertFalse( $results->getMetadata()->getCachingMetadata()->isCached() );
	}

	public function testGetResults_FutureDateStillInFuture() {
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [] );
		$resultsCache = new ResultsCache( new WANObjectCache( [ 'cache' => new HashBagOStuff() ] ), 'v2' );
		$cachingResultsSource = $this->getCachingResultsSource(
			$this->createMock( ResultsSource::class ),
			$resultsCache,
			$metaDataAccessor
		);

		$q5 = new ItemId( 'Q5' );
		$futureTimestamp = '+20018-02-16T00:00:00Z';
		$value = [
			'results' => [ [ 'CheckResult for Q5' ] ],
			'latestRevisionIds' => [],
			'futureTime' => [
				'time' => $futureTimestamp,
				'timezone' => 0,
				'before' => 0,
				'after' => 0,
				'precision' => TimeValue::PRECISION_DAY,
				'calendarmodel' => TimeValue::CALENDAR_GREGORIAN,
			],
		];
		$resultsCache->set( $q5, $value );

		$response = $cachingResultsSource->getStoredResults( $q5 );

		$this->assertNotNull( $response );
		$this->assertEquals( [ $this->getCheckResult( 'Q5' ) ], $response->getArray() );
		$dependencyMetadata = $response->getMetadata()->getDependencyMetadata();
		$this->assertNotNull( $dependencyMetadata->getFutureTime() );
		$this->assertSame( $futureTimestamp, $dependencyMetadata->getFutureTime()->getTime() );
	}

	public function testGetResults_FutureDateNowInPast() {
		$metaDataAccessor = $this->createMock( WikiPageEntityMetaDataAccessor::class );
		$metaDataAccessor->method( 'loadLatestRevisionIds' )
			->willReturn( [] );
		$resultsCache = new ResultsCache( new WANObjectCache( [ 'cache' => new HashBagOStuff() ] ), 'v2' );
		$cachingResultsSource = $this->getCachingResultsSource(
			$this->createMock( ResultsSource::class ),
			$resultsCache,
			$metaDataAccessor
		);

		$q5 = new ItemId( 'Q5' );
		$pastTimestamp = '+2018-02-14T00:00:00Z';
		$value = [
			'results' => [ [ 'CheckResult for Q5' ] ],
			'latestRevisionIds' => [],
			'futureTime' => [
				'time' => $pastTimestamp,
				'timezone' => 0,
				'before' => 0,
				'after' => 0,
				'precision' => TimeValue::PRECISION_DAY,
				'calendarmodel' => TimeValue::CALENDAR_GREGORIAN,
			],
		];
		$resultsCache->set( $q5, $value );

		$response = $cachingResultsSource->getStoredResults( $q5 );

		$this->assertNull( $response );
	}

	public function testGetResults_ConstraintIds() {
		$entityIds = [ new ItemId( 'Q5' ), new ItemId( 'Q10' ) ];
		$statementIds = [];
		$constraintIds = [ 'P12$11a14ea5-10dc-425b-b94d-6e65997be983' ];
		$statuses = [ 'garbage status', 'other status' ];
		$expected = new CachedCheckResults(
			[ $this->getCheckResult( 'Q5' ), $this->getCheckResult( 'Q10' ) ],
			Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 5 * 60 ) )
		);
		$cachingResultsSource = $this->getCachingResultsSourceMock(
			$this->createMock( ResultsSource::class ),
			new ResultsCache( WANObjectCache::newEmpty(), 'v2' ),
			$this->createMock( WikiPageEntityMetaDataAccessor::class )
		);
		$cachingResultsSource->expects( $this->never() )->method( 'getStoredResults' );
		$cachingResultsSource->expects( $this->once() )->method( 'getAndStoreResults' )
			->with( $entityIds, $statementIds, $constraintIds, $statuses )
			->willReturn( $expected );
		/** @var CachingResultsSource $cachingResultsSource */

		$results = $cachingResultsSource->getResults(
			$entityIds,
			$statementIds,
			$constraintIds,
			$statuses
		);

		$this->assertSame( $expected->getArray(), $results->getArray() );
		$this->assertEquals( $expected->getMetadata(), $results->getMetadata() );
	}

	public function testGetResults_StatementIds() {
		$entityIds = [];
		$statementIds = [ 'Q5$9c009c6f-fdf5-41d1-86e9-e790427e3dc6' ];
		$constraintIds = [];
		$statuses = [ 'garbage status', 'other status' ];
		$expected = new CachedCheckResults(
			[ $this->getCheckResult( 'Q5' ) ],
			Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 5 * 60 ) )
		);
		$cachingResultsSource = $this->getCachingResultsSourceMock(
			$this->createMock( ResultsSource::class ),
			new ResultsCache( WANObjectCache::newEmpty(), 'v2' ),
			$this->createMock( WikiPageEntityMetaDataAccessor::class )
		);
		$cachingResultsSource->expects( $this->never() )->method( 'getStoredResults' );
		$cachingResultsSource->expects( $this->once() )->method( 'getAndStoreResults' )
			->with( $entityIds, $statementIds, $constraintIds, $statuses )
			->willReturn( $expected );
		/** @var CachingResultsSource $cachingResultsSource */

		$results = $cachingResultsSource->getResults(
			$entityIds,
			$statementIds,
			$constraintIds,
			$statuses
		);

		$this->assertSame( $expected->getArray(), $results->getArray() );
		$this->assertEquals( $expected->getMetadata(), $results->getMetadata() );
	}

	public function testGetResults_FullyCached() {
		$entityIds = [ new ItemId( 'Q5' ), new ItemId( 'Q10' ) ];
		$statuses = CachingResultsSource::CACHED_STATUSES;
		$cachingResultsSource = $this->getCachingResultsSourceMock(
			$this->createMock( ResultsSource::class ),
			new ResultsCache( WANObjectCache::newEmpty(), 'v2' ),
			$this->createMock( WikiPageEntityMetaDataAccessor::class )
		);
		$cachingResultsSource->expects( $this->exactly( 2 ) )->method( 'getStoredResults' )
			->willReturnCallback( function ( EntityId $entityId ) {
				$serialization = $entityId->getSerialization();
				return new CachedCheckResults(
					[ $this->getCheckResult( $serialization ) ],
					Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 64800 ) )
				);
			} );
		$cachingResultsSource->expects( $this->never() )->method( 'getAndStoreResults' );
		/** @var CachingResultsSource $cachingResultsSource */

		$results = $cachingResultsSource->getResults(
			$entityIds,
			[],
			null,
			$statuses
		);

		$expected = [ $this->getCheckResult( 'Q5' ), $this->getCheckResult( 'Q10' ) ];
		$actual = $results->getArray();
		sort( $expected );
		sort( $actual );
		$this->assertEquals( $expected, $actual );
		$this->assertSame( 64800, $results->getMetadata()->getCachingMetadata()->getMaximumAgeInSeconds() );
	}

	public function testGetResults_PartiallyCached() {
		$entityIds = [ new ItemId( 'Q5' ), new ItemId( 'Q10' ) ];
		$statuses = CachingResultsSource::CACHED_STATUSES;
		$cachingResultsSource = $this->getCachingResultsSourceMock(
			$this->createMock( ResultsSource::class ),
			new ResultsCache( WANObjectCache::newEmpty(), 'v2' ),
			$this->createMock( WikiPageEntityMetaDataAccessor::class )
		);
		$cachingResultsSource->expects( $this->exactly( 2 ) )->method( 'getStoredResults' )
			->willReturnCallback( function ( EntityId $entityId ) {
				if ( $entityId->getSerialization() === 'Q5' ) {
					return new CachedCheckResults(
						[ $this->getCheckResult( 'Q5' ) ],
						Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 64800 ) )
					);
				} else {
					return null;
				}
			} );
		$cachingResultsSource->expects( $this->once() )->method( 'getAndStoreResults' )
			->with( [ $entityIds[1] ], [], null, $statuses )
			->willReturn( new CachedCheckResults(
				[ $this->getCheckResult( 'Q10' ) ],
				Metadata::ofDependencyMetadata( DependencyMetadata::ofEntityId( $entityIds[1] ) )
			) );
		/** @var CachingResultsSource $cachingResultsSource */

		$results = $cachingResultsSource->getResults(
			$entityIds,
			[],
			null,
			$statuses
		);

		$expected = [ $this->getCheckResult( 'Q5' ), $this->getCheckResult( 'Q10' ) ];
		$actual = $results->getArray();
		sort( $expected );
		sort( $actual );
		$this->assertEquals( $expected, $actual );
		$metadata = $results->getMetadata();
		$this->assertSame( 64800, $metadata->getCachingMetadata()->getMaximumAgeInSeconds() );
		$this->assertSame( [ $entityIds[1] ], $metadata->getDependencyMetadata()->getEntityIds() );
	}

	public function testGetResults_UseCacheWithMissingStatuses() {
		$entityIds = [ new ItemId( 'Q5' ) ];
		$statuses = array_values( array_diff(
			CachingResultsSource::CACHED_STATUSES,
			[ CheckResult::STATUS_WARNING ]
		) );
		$allResults = new CachedCheckResults(
			[
				$this->getCheckResult( 'Q5' ),
				$this->getCheckResult( 'Q10', CheckResult::STATUS_WARNING ),
			],
			Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 5 * 60 ) )
		);
		$cachingResultsSource = $this->getCachingResultsSourceMock(
			$this->createMock( ResultsSource::class ),
			new ResultsCache( WANObjectCache::newEmpty(), 'v2' ),
			$this->createMock( WikiPageEntityMetaDataAccessor::class )
		);
		$cachingResultsSource->expects( $this->once() )->method( 'getStoredResults' )
			->willReturn( $allResults );
		$cachingResultsSource->expects( $this->never() )->method( 'getAndStoreResults' );
		/** @var CachingResultsSource $cachingResultsSource */

		$results = $cachingResultsSource->getResults(
			$entityIds,
			[],
			null,
			$statuses
		);

		$expected = [ $allResults->getArray()[0] ]; // without [1], which has a non-requested status
		$actual = $results->getArray();
		sort( $expected );
		sort( $actual );
		$this->assertSame( $expected, $actual );
	}

	public function testGetResults_SkipCacheWithExtraStatuses() {
		$entityIds = [ new ItemId( 'Q5' ), new ItemId( 'Q10' ) ];
		$statuses = array_merge(
			CachingResultsSource::CACHED_STATUSES,
			[ CheckResult::STATUS_TODO ]
		);
		$results = new CachedCheckResults(
			[ $this->getCheckResult( 'Q5' ) ],
			Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 5 * 60 ) )
		);
		$cachingResultsSource = $this->getCachingResultsSourceMock(
			$this->createMock( ResultsSource::class ),
			new ResultsCache( WANObjectCache::newEmpty(), 'v2' ),
			$this->createMock( WikiPageEntityMetaDataAccessor::class )
		);
		$cachingResultsSource->expects( $this->never() )->method( 'getStoredResults' );
		$cachingResultsSource->expects( $this->once() )->method( 'getAndStoreResults' )
			->with( $entityIds, [], null, $statuses )
			->willReturn( $results );
		/** @var CachingResultsSource $cachingResultsSource */

		$cachingResultsSource->getResults(
			$entityIds,
			[],
			null,
			$statuses
		);
	}

}
