<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\PropertyId;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Statement\StatementList;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;

/**
 * Class for helper functions for the connection checkers.
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class ConnectionCheckerHelper {

	/**
	 * Finds a statement with the given property ID.
	 *
	 * @param StatementList $statementList
	 * @param PropertyId $propertyId
	 *
	 * @return Statement|null
	 */
	public function findStatementWithProperty(
		StatementList $statementList,
		PropertyId $propertyId
	) {
		$statementListByPropertyId = $statementList->getByPropertyId( $propertyId );
		if ( $statementListByPropertyId->isEmpty() ) {
			return null;
		} else {
			return $statementListByPropertyId->toArray()[0];
		}
	}

	/**
	 * Finds a statement with the given property ID and entity ID value.
	 *
	 * @param StatementList $statementList
	 * @param PropertyId $propertyId
	 * @param EntityId $value
	 *
	 * @return Statement|null
	 */
	public function findStatementWithPropertyAndEntityIdValue(
		StatementList $statementList,
		PropertyId $propertyId,
		EntityId $value
	) {
		$statementListByPropertyId = $statementList->getByPropertyId( $propertyId );
		/** @var Statement $statement */
		foreach ( $statementListByPropertyId as $statement ) {
			$snak = $statement->getMainSnak();
			if ( $snak instanceof PropertyValueSnak ) {
				$dataValue = $snak->getDataValue();
				if ( $dataValue instanceof EntityIdValue &&
					$dataValue->getEntityId()->equals( $value )
				) {
					return $statement;
				}
			}
		}
		return null;
	}

	/**
	 * Finds a statement with the given property ID and one of the given item ID snak values.
	 *
	 * @param StatementList $statementList
	 * @param PropertyId $propertyId
	 * @param ItemIdSnakValue[] $values
	 *
	 * @return Statement|null
	 */
	public function findStatementWithPropertyAndItemIdSnakValues(
		StatementList $statementList,
		PropertyId $propertyId,
		array $values
	) {
		$statementListByPropertyId = $statementList->getByPropertyId( $propertyId );
		/** @var Statement $statement */
		foreach ( $statementListByPropertyId as $statement ) {
			$snak = $statement->getMainSnak();
			foreach ( $values as $value ) {
				if ( $value->matchesSnak( $snak ) ) {
					return $statement;
				}
			}
		}
		return null;
	}

}
