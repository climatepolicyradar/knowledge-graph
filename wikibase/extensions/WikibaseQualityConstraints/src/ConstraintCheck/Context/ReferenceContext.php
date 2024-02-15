<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Context;

use Wikibase\DataModel\Entity\StatementListProvidingEntity;
use Wikibase\DataModel\Reference;
use Wikibase\DataModel\Snak\Snak;
use Wikibase\DataModel\Statement\Statement;

/**
 * A constraint check context for a snak of a reference of a statement.
 *
 * @license GPL-2.0-or-later
 */
class ReferenceContext extends AbstractContext {

	private Statement $statement;

	private Reference $reference;

	public function __construct(
		StatementListProvidingEntity $entity,
		Statement $statement,
		Reference $reference,
		Snak $snak
	) {
		parent::__construct( $entity, $snak );
		$this->statement = $statement;
		$this->reference = $reference;
	}

	public function getType(): string {
		return self::TYPE_REFERENCE;
	}

	public function getSnakGroup( string $groupingMode, array $separators = [] ): array {
		$snaks = $this->reference->getSnaks();
		return array_values( $snaks->getArrayCopy() );
	}

	public function getCursor(): ContextCursor {
		return new ReferenceContextCursor(
			$this->entity->getId()->getSerialization(),
			$this->statement->getPropertyId()->getSerialization(),
			$this->getStatementGuid( $this->statement ),
			$this->snak->getHash(),
			$this->snak->getPropertyId()->getSerialization(),
			$this->reference->getHash()
		);
	}

}
