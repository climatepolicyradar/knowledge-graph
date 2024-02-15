<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Context;

/**
 * @license GPL-2.0-or-later
 */
class QualifierContextCursor extends ApiV2ContextCursor {

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
	 * @param string $entityId
	 * @param string $statementPropertyId
	 * @param string $statementGuid
	 * @param string $snakHash
	 * @param string $snakPropertyId
	 */
	public function __construct(
		$entityId,
		$statementPropertyId,
		$statementGuid,
		$snakHash,
		$snakPropertyId
	) {
		$this->entityId = $entityId;
		$this->statementPropertyId = $statementPropertyId;
		$this->statementGuid = $statementGuid;
		$this->snakHash = $snakHash;
		$this->snakPropertyId = $snakPropertyId;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getType() {
		return Context::TYPE_QUALIFIER;
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

	protected function &getMainArray( array &$container ) {
		$statementArray = &$this->getStatementArray( $container );

		if ( !array_key_exists( 'qualifiers', $statementArray ) ) {
			$statementArray['qualifiers'] = [];
		}
		$qualifiersArray = &$statementArray['qualifiers'];

		$snakPropertyId = $this->getSnakPropertyId();
		if ( !array_key_exists( $snakPropertyId, $qualifiersArray ) ) {
			$qualifiersArray[$snakPropertyId] = [];
		}
		$propertyArray = &$qualifiersArray[$snakPropertyId];

		$snakHash = $this->getSnakHash();
		foreach ( $propertyArray as &$potentialQualifierArray ) {
			if ( $potentialQualifierArray['hash'] === $snakHash ) {
				$qualifierArray = &$potentialQualifierArray;
				break;
			}
		}
		if ( !isset( $qualifierArray ) ) {
			$qualifierArray = [ 'hash' => $snakHash ];
			$propertyArray[] = &$qualifierArray;
		}

		return $qualifierArray;
	}

}
