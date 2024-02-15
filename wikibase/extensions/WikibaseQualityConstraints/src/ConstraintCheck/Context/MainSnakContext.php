<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Context;

use LogicException;
use Wikibase\DataModel\Entity\PropertyId;
use Wikibase\DataModel\Entity\StatementListProvidingEntity;
use Wikibase\DataModel\Snak\Snak;
use Wikibase\DataModel\Snak\SnakList;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Statement\StatementList;

/**
 * A constraint check context for the main snak of a statement.
 *
 * @license GPL-2.0-or-later
 */
class MainSnakContext extends AbstractContext {

	private Statement $statement;

	public function __construct( StatementListProvidingEntity $entity, Statement $statement ) {
		parent::__construct( $entity, $statement->getMainSnak() );

		$this->statement = $statement;
	}

	public function getType(): string {
		return self::TYPE_STATEMENT;
	}

	public function getSnakRank(): ?int {
		return $this->statement->getRank();
	}

	public function getSnakStatement(): Statement {
		return $this->statement;
	}

	public function getSnakGroup( string $groupingMode, array $separators = [] ): array {
		/** @var StatementList $statements */
		$statements = $this->entity->getStatements();
		switch ( $groupingMode ) {
			case Context::GROUP_NON_DEPRECATED:
				$statements = $statements->getByRank( [
					Statement::RANK_NORMAL,
					Statement::RANK_PREFERRED,
				] );
				break;
			case Context::GROUP_BEST_RANK:
				$statements = $this->getBestStatementsPerPropertyId( $statements );
				break;
			default:
				throw new LogicException( 'Unknown $groupingMode ' . $groupingMode );
		}
		return $this->getStatementsWithSameQualifiersForProperties(
			$this->statement,
			$statements,
			$separators
		)->getMainSnaks();
	}

	private function getBestStatementsPerPropertyId( StatementList $statements ): StatementList {
		$allBestStatements = new StatementList();
		foreach ( $statements->getPropertyIds() as $propertyId ) {
			$bestStatements = $statements->getByPropertyId( $propertyId )
				->getBestStatements();
			foreach ( $bestStatements as $bestStatement ) {
				$allBestStatements->addStatement( $bestStatement );
			}
		}
		return $allBestStatements;
	}

	/**
	 * Returns the statements of a statement list
	 * which for a set of propert IDs have the same qualifiers as a certain statement.
	 * “unknown value” qualifiers are considered different from each other.
	 *
	 * @param Statement $currentStatement
	 * @param StatementList $allStatements
	 * @param PropertyId[] $qualifierPropertyIds
	 * @return StatementList
	 */
	private function getStatementsWithSameQualifiersForProperties(
		Statement $currentStatement,
		StatementList $allStatements,
		array $qualifierPropertyIds
	): StatementList {
		$similarStatements = new StatementList();
		foreach ( $allStatements as $statement ) {
			if ( $statement === $currentStatement ) {
				// if the statement has an “unknown value” qualifier,
				// it might be considered different from itself,
				// so add it explicitly to ensure it’s always included
				$similarStatements->addStatement( $statement );
				continue;
			}
			foreach ( $qualifierPropertyIds as $qualifierPropertyId ) {
				if ( !$this->haveSameQualifiers( $currentStatement, $statement, $qualifierPropertyId ) ) {
					continue 2;
				}
			}
			$similarStatements->addStatement( $statement );
		}
		return $similarStatements;
	}

	/**
	 * Tests whether two statements have the same qualifiers with a certain property ID.
	 * “unknown value” qualifiers are considered different from each other.
	 */
	private function haveSameQualifiers( Statement $s1, Statement $s2, PropertyId $propertyId ): bool {
		$q1 = $this->getSnaksWithPropertyId( $s1->getQualifiers(), $propertyId );
		$q2 = $this->getSnaksWithPropertyId( $s2->getQualifiers(), $propertyId );

		if ( $q1->count() !== $q2->count() ) {
			return false;
		}

		foreach ( $q1 as $qualifier ) {
			switch ( $qualifier->getType() ) {
				case 'value':
				case 'novalue':
					if ( !$q2->hasSnak( $qualifier ) ) {
						return false;
					}
					break;
				case 'somevalue':
					return false; // all “unknown value”s are considered different from each other
			}
		}

		// a SnakList cannot contain the same snak more than once,
		// so if every snak of q1 is also in q2 and their cardinality is identical,
		// then they must be entirely identical
		return true;
	}

	/**
	 * Returns the snaks of the given snak list with the specified property ID.
	 */
	private function getSnaksWithPropertyId( SnakList $allSnaks, PropertyId $propertyId ): SnakList {
		$snaks = new SnakList();
		/** @var Snak $snak */
		foreach ( $allSnaks as $snak ) {
			if ( $snak->getPropertyId()->equals( $propertyId ) ) {
				$snaks->addSnak( $snak );
			}
		}
		return $snaks;
	}

	public function getCursor(): ContextCursor {
		return new MainSnakContextCursor(
			$this->entity->getId()->getSerialization(),
			$this->statement->getPropertyId()->getSerialization(),
			$this->getStatementGuid( $this->statement ),
			$this->statement->getMainSnak()->getHash()
		);
	}

}
