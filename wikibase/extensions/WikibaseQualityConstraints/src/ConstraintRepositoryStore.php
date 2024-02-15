<?php

namespace WikibaseQuality\ConstraintReport;

use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikimedia\Rdbms\DBUnexpectedError;
use Wikimedia\Rdbms\ILoadBalancer;

/**
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class ConstraintRepositoryStore implements ConstraintStore {

	/** @var ILoadBalancer */
	private $lb;

	/** @var string|false */
	private $dbName;

	/**
	 * @param ILoadBalancer $lb Load balancer for database connections.
	 * Must match $dbName, i.e. if $dbName is not false,
	 * then using the main DBLoadBalancer service may be incorrect.
	 * @param string|false $dbName Database name ($domain for ILoadBalancer methods).
	 */
	public function __construct( ILoadBalancer $lb, $dbName ) {
		$this->lb = $lb;
		$this->dbName = $dbName;
	}

	private function encodeConstraintParameters( array $constraintParameters ) {
		$json = json_encode( $constraintParameters, JSON_FORCE_OBJECT );

		if ( strlen( $json ) > 50000 ) {
			$json = json_encode( [ '@error' => [ 'toolong' => true ] ] );
		}

		return $json;
	}

	/**
	 * @param Constraint[] $constraints
	 *
	 * @throws DBUnexpectedError
	 * @return bool
	 */
	public function insertBatch( array $constraints ) {
		$accumulator = array_map(
			function ( Constraint $constraint ) {
				return [
					'constraint_guid' => $constraint->getConstraintId(),
					'pid' => $constraint->getPropertyId()->getNumericId(),
					'constraint_type_qid' => $constraint->getConstraintTypeItemId(),
					'constraint_parameters' => $this->encodeConstraintParameters( $constraint->getConstraintParameters() ),
				];
			},
			$constraints
		);

		$dbw = $this->lb->getConnection( ILoadBalancer::DB_PRIMARY, [], $this->dbName );
		return $dbw->insert( 'wbqc_constraints', $accumulator, __METHOD__ );
	}

	/**
	 * Delete all constraints for the property ID.
	 *
	 * @param NumericPropertyId $propertyId
	 *
	 * @throws DBUnexpectedError
	 */
	public function deleteForProperty( NumericPropertyId $propertyId ) {
		$dbw = $this->lb->getConnection( ILoadBalancer::DB_PRIMARY, [], $this->dbName );
		$dbw->delete(
			'wbqc_constraints',
			[
				'pid' => $propertyId->getNumericId(),
			],
			__METHOD__
		);
	}

}
