<?php

namespace WikibaseQuality\ConstraintReport\Api;

use DataValues\TimeValue;
use WANObjectCache;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\EntityIdParser;
use Wikibase\Lib\Store\LookupConstants;
use Wikibase\Lib\Store\Sql\WikiPageEntityMetaDataAccessor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckResults;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachingMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\DependencyMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\LoggingHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\TimeValueComparer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\NullResult;

/**
 * A ResultsSource that wraps another ResultsSource,
 * storing results in a cache
 * and retrieving them from there if the results are still fresh.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CachingResultsSource implements ResultsSource {

	public const CACHED_STATUSES = [
		CheckResult::STATUS_VIOLATION,
		CheckResult::STATUS_WARNING,
		CheckResult::STATUS_SUGGESTION,
		CheckResult::STATUS_BAD_PARAMETERS,
	];

	/**
	 * @var ResultsSource
	 */
	private $resultsSource;

	/**
	 * @var ResultsCache
	 */
	private $cache;

	/**
	 * @var CheckResultSerializer
	 */
	private $checkResultSerializer;

	/**
	 * @var CheckResultDeserializer
	 */
	private $checkResultDeserializer;

	/**
	 * @var WikiPageEntityMetaDataAccessor
	 */
	private $wikiPageEntityMetaDataAccessor;

	/**
	 * @var EntityIdParser
	 */
	private $entityIdParser;

	/**
	 * @var int
	 */
	private $ttlInSeconds;

	/**
	 * @var string[]
	 */
	private $possiblyStaleConstraintTypes;

	/**
	 * @var int
	 */
	private $maxRevisionIds;

	/**
	 * @var LoggingHelper
	 */
	private $loggingHelper;

	/**
	 * @var TimeValueComparer
	 */
	private $timeValueComparer;

	/**
	 * @var callable
	 */
	private $microtime = 'microtime';

	/**
	 * @param ResultsSource $resultsSource The ResultsSource that cache misses are delegated to.
	 * @param ResultsCache $cache The cache where results can be stored.
	 * @param CheckResultSerializer $checkResultSerializer Used to serialize check results.
	 * @param CheckResultDeserializer $checkResultDeserializer Used to deserialize check results.
	 * @param WikiPageEntityMetaDataAccessor $wikiPageEntityMetaDataAccessor Used to get the latest revision ID.
	 * @param EntityIdParser $entityIdParser Used to parse entity IDs in cached objects.
	 * @param int $ttlInSeconds Time-to-live of the cached values, in seconds.
	 * @param string[] $possiblyStaleConstraintTypes item IDs of constraint types
	 * where cached results may always be stale, regardless of invalidation logic
	 * @param int $maxRevisionIds The maximum number of revision IDs to check;
	 * if a check result depends on more entity IDs than this number, it is not cached.
	 * @param LoggingHelper $loggingHelper
	 */
	public function __construct(
		ResultsSource $resultsSource,
		ResultsCache $cache,
		CheckResultSerializer $checkResultSerializer,
		CheckResultDeserializer $checkResultDeserializer,
		WikiPageEntityMetaDataAccessor $wikiPageEntityMetaDataAccessor,
		EntityIdParser $entityIdParser,
		$ttlInSeconds,
		array $possiblyStaleConstraintTypes,
		$maxRevisionIds,
		LoggingHelper $loggingHelper
	) {
		$this->resultsSource = $resultsSource;
		$this->cache = $cache;
		$this->checkResultSerializer = $checkResultSerializer;
		$this->checkResultDeserializer = $checkResultDeserializer;
		$this->wikiPageEntityMetaDataAccessor = $wikiPageEntityMetaDataAccessor;
		$this->entityIdParser = $entityIdParser;
		$this->ttlInSeconds = $ttlInSeconds;
		$this->possiblyStaleConstraintTypes = $possiblyStaleConstraintTypes;
		$this->maxRevisionIds = $maxRevisionIds;
		$this->loggingHelper = $loggingHelper;
		$this->timeValueComparer = new TimeValueComparer();
	}

	public function getResults(
		array $entityIds,
		array $claimIds,
		?array $constraintIds,
		array $statuses
	) {
		$results = [];
		$metadatas = [];
		if ( $this->canUseStoredResults( $entityIds, $claimIds, $constraintIds, $statuses ) ) {
			$storedEntityIds = [];
			foreach ( $entityIds as $entityId ) {
				$storedResults = $this->getStoredResults( $entityId );
				if ( $storedResults !== null ) {
					$this->loggingHelper->logCheckConstraintsCacheHit( $entityId );
					foreach ( $storedResults->getArray() as $checkResult ) {
						if ( $this->statusSelected( $statuses, $checkResult ) ) {
							$results[] = $checkResult;
						}
					}
					$metadatas[] = $storedResults->getMetadata();
					$storedEntityIds[] = $entityId;
				}
			}
			$entityIds = array_values( array_diff( $entityIds, $storedEntityIds ) );
		}
		if ( $entityIds !== [] || $claimIds !== [] ) {
			if ( $entityIds !== [] ) {
				$this->loggingHelper->logCheckConstraintsCacheMisses( $entityIds );
			}
			$response = $this->getAndStoreResults( $entityIds, $claimIds, $constraintIds, $statuses );
			$results = array_merge( $results, $response->getArray() );
			$metadatas[] = $response->getMetadata();
		}
		return new CachedCheckResults(
			$results,
			Metadata::merge( $metadatas )
		);
	}

	/**
	 * We can only use cached constraint results
	 * if nothing more than the problematic results of a full constraint check were requested:
	 * constraint checks for the full entity (not just individual statements),
	 * without restricting the set of constraints to check,
	 * and with no statuses other than 'violation', 'warning' and 'bad-parameters'.
	 *
	 * @param EntityId[] $entityIds
	 * @param string[] $claimIds
	 * @param string[]|null $constraintIds
	 * @param string[] $statuses
	 * @return bool
	 */
	private function canUseStoredResults(
		array $entityIds,
		array $claimIds,
		?array $constraintIds,
		array $statuses
	) {
		if ( $claimIds !== [] ) {
			return false;
		}
		if ( $constraintIds !== null ) {
			return false;
		}
		if ( array_diff( $statuses, self::CACHED_STATUSES ) !== [] ) {
			return false;
		}
		return true;
	}

	/**
	 * Check whether a check result should be used,
	 * either because it has the right status
	 * or because it is a NullResult whose metadata should be preserved.
	 *
	 * @param string[] $statuses
	 * @param CheckResult $result
	 * @return bool
	 */
	private function statusSelected( array $statuses, CheckResult $result ) {
		return in_array( $result->getStatus(), $statuses, true ) ||
			$result instanceof NullResult;
	}

	/**
	 * @param EntityId[] $entityIds
	 * @param string[] $claimIds
	 * @param string[]|null $constraintIds
	 * @param string[] $statuses
	 * @return CachedCheckResults
	 */
	public function getAndStoreResults(
		array $entityIds,
		array $claimIds,
		?array $constraintIds,
		array $statuses
	) {
		$results = $this->resultsSource->getResults( $entityIds, $claimIds, $constraintIds, $statuses );

		if ( $this->canStoreResults( $entityIds, $claimIds, $constraintIds, $statuses ) ) {
			foreach ( $entityIds as $entityId ) {
				$this->storeResults( $entityId, $results );
			}
		}

		return $results;
	}

	/**
	 * We can only store constraint results
	 * if the set of constraints to check was not restricted
	 * and all the problematic results were requested.
	 * However, it doesn’t matter whether constraint checks on individual statements were requested:
	 * we only store results for the mentioned entity IDs,
	 * and those will be complete regardless of what’s in the statement IDs.
	 * And it also doesn’t matter whether the set of statuses requested
	 * was exactly the statuses we cache or a superset of it:
	 * as long as all the results we want to cache are there,
	 * we can filter out the extraneous ones before we serialize them.
	 *
	 * @param EntityId[] $entityIds
	 * @param string[] $claimIds
	 * @param ?string[] $constraintIds
	 * @param string[] $statuses
	 * @return bool
	 */
	private function canStoreResults(
		array $entityIds,
		array $claimIds,
		?array $constraintIds,
		array $statuses
	) {
		if ( $constraintIds !== null ) {
			return false;
		}
		if ( array_diff( self::CACHED_STATUSES, $statuses ) !== [] ) {
			return false;
		}
		return true;
	}

	/**
	 * Store check results for the given entity ID in the cache, if possible.
	 *
	 * @param EntityId $entityId The entity ID.
	 * @param CachedCheckResults $results A collection of check results with metadata.
	 * May include check results for other entity IDs as well,
	 * or check results with statuses that we’re not interested in caching.
	 */
	private function storeResults( EntityId $entityId, CachedCheckResults $results ) {
		$latestRevisionIds = $this->getLatestRevisionIds(
			$results->getMetadata()->getDependencyMetadata()->getEntityIds()
		);
		if ( $latestRevisionIds === null ) {
			return;
		}

		$resultSerializations = [];
		foreach ( $results->getArray() as $checkResult ) {
			if ( $checkResult->getContextCursor()->getEntityId() !== $entityId->getSerialization() ) {
				continue;
			}
			if ( $this->statusSelected( self::CACHED_STATUSES, $checkResult ) ) {
				$resultSerializations[] = $this->checkResultSerializer->serialize( $checkResult );
			}
		}

		$value = [
			'results' => $resultSerializations,
			'latestRevisionIds' => $latestRevisionIds,
		];
		$futureTime = $results->getMetadata()->getDependencyMetadata()->getFutureTime();
		if ( $futureTime !== null ) {
			$value['futureTime'] = $futureTime->getArrayValue();
		}

		$this->cache->set( $entityId, $value, $this->ttlInSeconds );
	}

	/**
	 * @param EntityId $entityId
	 * @param int $forRevision Requested revision of $entityId
	 *            If this parameter is not zero, the results are returned if this is the latest revision,
	 *            otherwise null is returned, since we can't get constraints for past revisions.
	 * @return CachedCheckResults|null
	 */
	public function getStoredResults(
		EntityId $entityId,
		$forRevision = 0
	) {
		$cacheInfo = WANObjectCache::PASS_BY_REF;
		$value = $this->cache->get( $entityId, $curTTL, [], $cacheInfo );
		$now = call_user_func( $this->microtime, true );

		$dependencyMetadata = $this->checkDependencyMetadata( $value,
			[ $entityId->getSerialization() => $forRevision ] );
		if ( $dependencyMetadata === null ) {
			return null;
		}

		// @phan-suppress-next-line PhanTypePossiblyInvalidDimOffset False positive
		$asOf = $cacheInfo[WANObjectCache::KEY_AS_OF];
		$ageInSeconds = (int)ceil( $now - $asOf );
		$cachingMetadata = $ageInSeconds > 0 ?
			CachingMetadata::ofMaximumAgeInSeconds( $ageInSeconds ) :
			CachingMetadata::fresh();

		$results = [];
		foreach ( $value['results'] as $resultSerialization ) {
			$results[] = $this->deserializeCheckResult( $resultSerialization, $cachingMetadata );
		}

		return new CachedCheckResults(
			$results,
			Metadata::merge( [
				Metadata::ofCachingMetadata( $cachingMetadata ),
				Metadata::ofDependencyMetadata( $dependencyMetadata ),
			] )
		);
	}

	/**
	 * Extract the dependency metadata of $value
	 * and check that the dependency metadata does not indicate staleness.
	 *
	 * @param array|false $value
	 * @param int[] $paramRevs Revisions from parameters, id => revision
	 *   These revisions are used instead of ones recorded in the metadata,
	 *   so we can serve requests specifying concrete revisions, and if they are not latest,
	 *   we will reject then.
	 * @return DependencyMetadata|null the dependency metadata,
	 * or null if $value should no longer be used
	 */
	private function checkDependencyMetadata( $value, $paramRevs ) {
		if ( $value === false ) {
			return null;
		}

		if ( array_key_exists( 'futureTime', $value ) ) {
			$futureTime = TimeValue::newFromArray( $value['futureTime'] );
			if ( !$this->timeValueComparer->isFutureTime( $futureTime ) ) {
				return null;
			}
			$futureTimeDependencyMetadata = DependencyMetadata::ofFutureTime( $futureTime );
		} else {
			$futureTimeDependencyMetadata = DependencyMetadata::blank();
		}

		foreach ( $paramRevs as $id => $revision ) {
			if ( $revision > 0 ) {
				$value['latestRevisionIds'][$id] = min( $revision, $value['latestRevisionIds'][$id] ?? PHP_INT_MAX );
			}
		}

		$dependedEntityIds = array_map(
			[ $this->entityIdParser, "parse" ],
			array_keys( $value['latestRevisionIds'] )
		);

		if ( $value['latestRevisionIds'] !== $this->getLatestRevisionIds( $dependedEntityIds ) ) {
			return null;
		}

		return array_reduce(
			$dependedEntityIds,
			static function ( DependencyMetadata $metadata, EntityId $entityId ) {
				return DependencyMetadata::merge( [
					$metadata,
					DependencyMetadata::ofEntityId( $entityId ),
				] );
			},
			$futureTimeDependencyMetadata
		);
	}

	/**
	 * Deserialize a check result.
	 * If the result might be stale after caching
	 * (because its dependencies cannot be fully tracked in its dependency metadata),
	 * also add $cachingMetadata to it.
	 *
	 * @param array $resultSerialization
	 * @param CachingMetadata $cachingMetadata
	 * @return CheckResult
	 */
	private function deserializeCheckResult(
		array $resultSerialization,
		CachingMetadata $cachingMetadata
	) {
		$result = $this->checkResultDeserializer->deserialize( $resultSerialization );
		if ( $this->isPossiblyStaleResult( $result ) ) {
			$result->withMetadata(
				Metadata::merge( [
					$result->getMetadata(),
					Metadata::ofCachingMetadata( $cachingMetadata ),
				] )
			);
		}
		return $result;
	}

	/**
	 * @param CheckResult $result
	 * @return bool
	 */
	private function isPossiblyStaleResult( CheckResult $result ) {
		if ( $result instanceof NullResult ) {
			return false;
		}

		return in_array(
			$result->getConstraint()->getConstraintTypeItemId(),
			$this->possiblyStaleConstraintTypes
		);
	}

	/**
	 * @param EntityId[] $entityIds
	 * @return int[]|null array from entity ID serializations to revision ID,
	 * or null to indicate that not all revision IDs could be loaded
	 */
	private function getLatestRevisionIds( array $entityIds ) {
		if ( $entityIds === [] ) {
			$this->loggingHelper->logEmptyDependencyMetadata();
			return [];
		}
		if ( count( $entityIds ) > $this->maxRevisionIds ) {
			// one of those entities will probably be edited soon, so might as well skip caching
			$this->loggingHelper->logHugeDependencyMetadata( $entityIds, $this->maxRevisionIds );
			return null;
		}

		$latestRevisionIds = $this->wikiPageEntityMetaDataAccessor->loadLatestRevisionIds(
			$entityIds,
			LookupConstants::LATEST_FROM_REPLICA
		);
		if ( $this->hasFalseElements( $latestRevisionIds ) ) {
			return null;
		}
		return $latestRevisionIds;
	}

	/**
	 * @param array $array
	 * @return bool
	 */
	private function hasFalseElements( array $array ) {
		return in_array( false, $array, true );
	}

	/**
	 * Set a custom function to get the current time, instead of microtime().
	 *
	 * @param callable $microtime
	 */
	public function setMicrotimeFunction( callable $microtime ) {
		$this->microtime = $microtime;
	}

}
