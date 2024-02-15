<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Cache;

use DataValues\TimeValue;
use Wikibase\DataModel\Entity\EntityId;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\TimeValueComparer;
use Wikimedia\Assert\Assert;

/**
 * Information about what other things a value depends on.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class DependencyMetadata {

	/**
	 * @var EntityId[]
	 */
	private $entityIds = [];

	/**
	 * @var TimeValue|null
	 */
	private $timeValue = null;

	/**
	 * @return self Indication that a value does not depend on anything else.
	 */
	public static function blank() {
		return new self;
	}

	/**
	 * Track that a value depends on a certain entity ID.
	 * @param EntityId $entityId An entity ID from which the value was derived.
	 * @return self Indication that a value was derived from the entity with the given ID.
	 */
	public static function ofEntityId( EntityId $entityId ) {
		$ret = new self;
		$ret->entityIds[] = $entityId;
		return $ret;
	}

	/**
	 * Track that a value depends on a certain time value being a future date, not a past one.
	 * @param TimeValue $timeValue A point in time on which the value might start to become invalid.
	 * @return self Indication that a value will only remain valid
	 * as long as the given time value is in the future, not in the past.
	 */
	public static function ofFutureTime( TimeValue $timeValue ) {
		$ret = new self;
		$ret->timeValue = $timeValue;
		return $ret;
	}

	/**
	 * @param self[] $metadatas
	 * @return self
	 */
	public static function merge( array $metadatas ) {
		Assert::parameterElementType( self::class, $metadatas, '$metadatas' );
		$ret = new self;
		$entityIds = [];
		foreach ( $metadatas as $metadata ) {
			foreach ( $metadata->entityIds as $entityId ) {
				$entityIds[$entityId->getSerialization()] = $entityId;
			}
			$ret->timeValue = self::minTimeValue( $ret->timeValue, $metadata->timeValue );
		}
		$ret->entityIds = array_values( $entityIds );
		return $ret;
	}

	/**
	 * @param TimeValue|null $t1
	 * @param TimeValue|null $t2
	 * @return TimeValue|null
	 */
	private static function minTimeValue( TimeValue $t1 = null, TimeValue $t2 = null ) {
		if ( $t1 === null ) {
			return $t2;
		}
		if ( $t2 === null ) {
			return $t1;
		}
		return ( new TimeValueComparer() )->getMinimum( $t1, $t2 );
	}

	/**
	 * @return EntityId[] Entity IDs from whose entities the value was derived.
	 * Changes to those entity IDs should invalidate the value.
	 */
	public function getEntityIds() {
		return $this->entityIds;
	}

	/**
	 * @return TimeValue|null A point in time which was in the future when the value was first determined.
	 * The value should be invalidated once this point in time is no longer in the future.
	 */
	public function getFutureTime() {
		return $this->timeValue;
	}

}
