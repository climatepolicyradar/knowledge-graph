<?php

namespace WikibaseQuality\ConstraintReport\Api;

use Wikibase\DataModel\Entity\EntityId;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckResults;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\DependencyMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\EntityContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\DelegatingConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\NullResult;

/**
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CheckingResultsSource implements ResultsSource {

	/**
	 * @var DelegatingConstraintChecker
	 */
	private $delegatingConstraintChecker;

	public function __construct(
		DelegatingConstraintChecker $delegatingConstraintChecker
	) {
		$this->delegatingConstraintChecker = $delegatingConstraintChecker;
	}

	/**
	 * @param EntityId[] $entityIds
	 * @param string[] $claimIds
	 * @param ?string[] $constraintIds
	 * @param string[] $statuses
	 * @return CachedCheckResults
	 */
	public function getResults(
		array $entityIds,
		array $claimIds,
		?array $constraintIds,
		array $statuses
	) {
		$results = [];
		$metadatas = [];
		$statusesFlipped = array_flip( $statuses );
		foreach ( $entityIds as $entityId ) {
			$entityResults = $this->delegatingConstraintChecker->checkAgainstConstraintsOnEntityId(
				$entityId,
				$constraintIds,
				[ $this, 'defaultResultsPerContext' ],
				[ $this, 'defaultResultsPerEntity' ]
			);
			foreach ( $entityResults as $result ) {
				$metadatas[] = $result->getMetadata();
				if ( $this->statusSelected( $statusesFlipped, $result ) ) {
					$results[] = $result;
				}
			}
		}
		foreach ( $claimIds as $claimId ) {
			$claimResults = $this->delegatingConstraintChecker->checkAgainstConstraintsOnClaimId(
				$claimId,
				$constraintIds,
				[ $this, 'defaultResultsPerContext' ]
			);
			foreach ( $claimResults as $result ) {
				$metadatas[] = $result->getMetadata();
				if ( $this->statusSelected( $statusesFlipped, $result ) ) {
					$results[] = $result;
				}
			}
		}
		return new CachedCheckResults(
			$results,
			Metadata::merge( $metadatas )
		);
	}

	public function defaultResultsPerContext( Context $context ) {
		return $context->getType() === Context::TYPE_STATEMENT ?
			[ new NullResult( $context->getCursor() ) ] :
			[];
	}

	public function defaultResultsPerEntity( EntityId $entityId ) {
		return [
			( new NullResult( new EntityContextCursor( $entityId->getSerialization() ) ) )
				->withMetadata( Metadata::ofDependencyMetadata(
					DependencyMetadata::ofEntityId( $entityId )
				) ),
		];
	}

	public function statusSelected( array $statusesFlipped, CheckResult $result ) {
		return array_key_exists( $result->getStatus(), $statusesFlipped ) ||
			$result instanceof NullResult;
	}

}
