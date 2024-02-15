<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Context;

/**
 * @license GPL-2.0-or-later
 */
class ReferenceContextCursor extends ApiV2ContextCursor {

	/**
	 * @var string
	 */
	private $entityId;

	/**
	 * @var string
	 */
	private $statementPropertyId;

	/**
	 * @var string
	 */
	private $statementGuid;

	/**
	 * @var string
	 */
	private $snakHash;

	/**
	 * @var string
	 */
	private $snakPropertyId;

	/**
	 * @var string
	 */
	private $referenceHash;

	/**
	 * @param string $entityId
	 * @param string $statementPropertyId
	 * @param string $statementGuid
	 * @param string $snakHash
	 * @param string $snakPropertyId
	 * @param string $referenceHash
	 */
	public function __construct(
		$entityId,
		$statementPropertyId,
		$statementGuid,
		$snakHash,
		$snakPropertyId,
		$referenceHash
	) {
		$this->entityId = $entityId;
		$this->statementPropertyId = $statementPropertyId;
		$this->statementGuid = $statementGuid;
		$this->snakHash = $snakHash;
		$this->snakPropertyId = $snakPropertyId;
		$this->referenceHash = $referenceHash;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getType() {
		return Context::TYPE_REFERENCE;
	}

	public function getEntityId() {
		return $this->entityId;
	}

	public function getStatementPropertyId() {
		return $this->statementPropertyId;
	}

	public function getStatementGuid() {
		return $this->statementGuid;
	}

	public function getSnakPropertyId() {
		return $this->snakPropertyId;
	}

	public function getSnakHash() {
		return $this->snakHash;
	}

	public function getReferenceHash() {
		return $this->referenceHash;
	}

	protected function &getMainArray( array &$container ) {
		$statementArray = &$this->getStatementArray( $container );

		if ( !array_key_exists( 'references', $statementArray ) ) {
			$statementArray['references'] = [];
		}
		$referencesArray = &$statementArray['references'];

		$referenceHash = $this->getReferenceHash();
		foreach ( $referencesArray as &$potentialReferenceArray ) {
			if ( $potentialReferenceArray['hash'] === $referenceHash ) {
				$referenceArray = &$potentialReferenceArray;
				break;
			}
		}
		if ( !isset( $referenceArray ) ) {
			$referenceArray = [ 'hash' => $referenceHash, 'snaks' => [] ];
			$referencesArray[] = &$referenceArray;
		}

		$snaksArray = &$referenceArray['snaks'];

		$snakPropertyId = $this->getSnakPropertyId();
		if ( !array_key_exists( $snakPropertyId, $snaksArray ) ) {
			$snaksArray[$snakPropertyId] = [];
		}
		$propertyArray = &$snaksArray[$snakPropertyId];

		$snakHash = $this->getSnakHash();
		foreach ( $propertyArray as &$potentialSnakArray ) {
			if ( $potentialSnakArray['hash'] === $snakHash ) {
				$snakArray = &$potentialSnakArray;
				break;
			}
		}
		if ( !isset( $snakArray ) ) {
			$snakArray = [ 'hash' => $snakHash ];
			$propertyArray[] = &$snakArray;
		}

		return $snakArray;
	}

}
