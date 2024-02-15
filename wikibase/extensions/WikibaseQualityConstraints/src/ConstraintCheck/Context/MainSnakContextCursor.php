<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Context;

/**
 * @license GPL-2.0-or-later
 */
class MainSnakContextCursor extends ApiV2ContextCursor {

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
	 * @param string $entityId
	 * @param string $statementPropertyId
	 * @param string $statementGuid
	 * @param string $snakHash
	 */
	public function __construct(
		$entityId,
		$statementPropertyId,
		$statementGuid,
		$snakHash
	) {
		$this->entityId = $entityId;
		$this->statementPropertyId = $statementPropertyId;
		$this->statementGuid = $statementGuid;
		$this->snakHash = $snakHash;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getType() {
		return Context::TYPE_STATEMENT;
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
		return $this->statementPropertyId;
	}

	public function getSnakHash() {
		return $this->snakHash;
	}

	protected function &getMainArray( array &$container ) {
		$statementArray = &$this->getStatementArray( $container );

		if ( !array_key_exists( 'mainsnak', $statementArray ) ) {
			$snakHash = $this->getSnakHash();
			$statementArray['mainsnak'] = [ 'hash' => $snakHash ];
		}
		$mainsnakArray = &$statementArray['mainsnak'];

		return $mainsnakArray;
	}

}
