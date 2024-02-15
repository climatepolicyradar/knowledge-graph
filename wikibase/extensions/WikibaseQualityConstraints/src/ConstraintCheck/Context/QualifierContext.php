<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Context;

use Wikibase\DataModel\Entity\StatementListProvidingEntity;
use Wikibase\DataModel\Snak\Snak;
use Wikibase\DataModel\Statement\Statement;

/**
 * A constraint check context for a qualifier of a statement.
 *
 * @license GPL-2.0-or-later
 */
class QualifierContext extends AbstractContext {

	private Statement $statement;

	public function __construct(
		StatementListProvidingEntity $entity,
		Statement $statement,
		Snak $snak
	) {
		parent::__construct( $entity, $snak );
		$this->statement = $statement;
	}

	public function getType(): string {
		return self::TYPE_QUALIFIER;
	}

	public function getSnakGroup( string $groupingMode, array $separators = [] ): array {
		$snaks = $this->statement->getQualifiers();
		return array_values( $snaks->getArrayCopy() );
	}

	public function getCursor(): ContextCursor {
		return new QualifierContextCursor(
			$this->entity->getId()->getSerialization(),
			$this->statement->getPropertyId()->getSerialization(),
			$this->getStatementGuid( $this->statement ),
			$this->snak->getHash(),
			$this->snak->getPropertyId()->getSerialization()
		);
	}

}
