<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Context;

use InvalidArgumentException;
use Wikibase\DataModel\Entity\StatementListProvidingEntity;
use Wikibase\DataModel\Snak\Snak;
use Wikibase\DataModel\Statement\Statement;

/**
 * Base implementation of some Context functions,
 * given a snak and an entity.
 *
 * @license GPL-2.0-or-later
 */
abstract class AbstractContext implements Context {

	protected StatementListProvidingEntity $entity;

	protected Snak $snak;

	public function __construct( StatementListProvidingEntity $entity, Snak $snak ) {
		$this->entity = $entity;
		$this->snak = $snak;
	}

	public function getSnak(): Snak {
		return $this->snak;
	}

	public function getEntity(): StatementListProvidingEntity {
		return $this->entity;
	}

	// unimplemented: getType

	public function getSnakRank(): ?int {
		return null;
	}

	public function getSnakStatement(): ?Statement {
		return null;
	}

	// unimplemented: getCursor

	/** Helper function for {@link getCursor()} implementations. */
	protected function getStatementGuid( Statement $statement ): string {
		$guid = $statement->getGuid();
		if ( $guid === null ) {
			if ( defined( 'MW_PHPUNIT_TEST' ) ) {
				// let unit tests get away with not specifying a statement GUID:
				// much more convenient to fake it here than to add one to all tests
				return 'Q0$DEADBEEF-DEAD-BEEF-DEAD-BEEFDEADBEEF';
			} else {
				throw new InvalidArgumentException( 'Statement for Context must have a GUID' );
			}
		}
		return $guid;
	}

}
