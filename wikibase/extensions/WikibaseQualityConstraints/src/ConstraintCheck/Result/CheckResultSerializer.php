<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Result;

use Wikibase\DataModel\Entity\EntityId;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachingMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursorSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer;
use WikibaseQuality\ConstraintReport\ConstraintSerializer;

/**
 * A serializer for {@link CheckResult}s.
 * Caching metadata for {@link NullResult}s is never serialized,
 * since it doesn’t make sense for those results to be cached
 * (though they can carry dependency metadata, which is serialized correctly).
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CheckResultSerializer {

	public const KEY_CONTEXT_CURSOR = '|';
	public const KEY_CONSTRAINT = 'c';
	public const KEY_CHECK_RESULT_STATUS = 's';
	public const KEY_VIOLATION_MESSAGE = 'm';
	public const KEY_CONSTRAINT_CLARIFICATION = 'l';
	public const KEY_CACHING_METADATA = 'CM';
	public const KEY_DEPENDENCY_METADATA = 'DM';
	public const KEY_NULL_RESULT = '0';

	public const KEY_CACHING_METADATA_MAX_AGE = 'a';

	public const KEY_DEPENDENCY_METADATA_ENTITY_IDS = 'e';
	public const KEY_DEPENDENCY_METADATA_FUTURE_TIME = 'f';

	private ConstraintSerializer $constraintSerializer;
	private ContextCursorSerializer $contextCursorSerializer;
	private ViolationMessageSerializer $violationMessageSerializer;
	private bool $serializeDependencyMetadata;

	/**
	 * @param ConstraintSerializer $constraintSerializer
	 * @param ContextCursorSerializer $contextCursorSerializer
	 * @param ViolationMessageSerializer $violationMessageSerializer
	 * @param bool $serializeDependencyMetadata Whether to serialize the DependencyMetadata component
	 * of a result’s {@link CheckResult::getMetadata metadata} or not.
	 */
	public function __construct(
		ConstraintSerializer $constraintSerializer,
		ContextCursorSerializer $contextCursorSerializer,
		ViolationMessageSerializer $violationMessageSerializer,
		bool $serializeDependencyMetadata = true
	) {
		$this->constraintSerializer = $constraintSerializer;
		$this->contextCursorSerializer = $contextCursorSerializer;
		$this->violationMessageSerializer = $violationMessageSerializer;
		$this->serializeDependencyMetadata = $serializeDependencyMetadata;
	}

	public function serialize( CheckResult $checkResult ): array {
		$contextCursor = $checkResult->getContextCursor();

		$serialization = [
			self::KEY_CONTEXT_CURSOR => $this->contextCursorSerializer->serialize( $contextCursor ),
		];

		if ( $checkResult instanceof NullResult ) {
			$serialization[self::KEY_NULL_RESULT] = 1;
		} else {
			$constraint = $checkResult->getConstraint();
			$cachingMetadata = $checkResult->getMetadata()->getCachingMetadata();
			$violationMessage = $checkResult->getMessage();
			$clarification = $checkResult->getConstraintClarification();

			$serialization[self::KEY_CONSTRAINT] =
				$this->constraintSerializer->serialize( $constraint );
			$serialization[self::KEY_CHECK_RESULT_STATUS] =
				$checkResult->getStatus();
			$serialization[self::KEY_CACHING_METADATA] =
				$this->serializeCachingMetadata( $cachingMetadata );

			if ( $violationMessage !== null ) {
				$serialization[self::KEY_VIOLATION_MESSAGE] =
					$this->violationMessageSerializer->serialize( $violationMessage );
			}
			if ( $clarification->getTexts() !== [] ) {
				$serialization[self::KEY_CONSTRAINT_CLARIFICATION] =
					$clarification->getArrayValue();
			}
		}

		if ( $this->serializeDependencyMetadata ) {
			$serialization[self::KEY_DEPENDENCY_METADATA] =
				$this->serializeDependencyMetadata( $checkResult );
		}

		return $serialization;
	}

	private function serializeCachingMetadata( CachingMetadata $cachingMetadata ): array {
		$maximumAge = $cachingMetadata->getMaximumAgeInSeconds();

		$serialization = [];

		if ( $maximumAge > 0 ) {
			$serialization[self::KEY_CACHING_METADATA_MAX_AGE] = $maximumAge;
		}

		return $serialization;
	}

	private function serializeDependencyMetadata( CheckResult $checkResult ): array {
		$dependencyMetadata = $checkResult->getMetadata()->getDependencyMetadata();
		$entityIds = $dependencyMetadata->getEntityIds();
		$futureTime = $dependencyMetadata->getFutureTime();

		$serialization = [
			self::KEY_DEPENDENCY_METADATA_ENTITY_IDS => array_map(
				static function ( EntityId $entityId ) {
					return $entityId->getSerialization();
				},
				$entityIds
			),
		];

		if ( $futureTime !== null ) {
			$serialization[self::KEY_DEPENDENCY_METADATA_FUTURE_TIME] =
				$futureTime->getArrayValue();
		}

		return $serialization;
	}

}
