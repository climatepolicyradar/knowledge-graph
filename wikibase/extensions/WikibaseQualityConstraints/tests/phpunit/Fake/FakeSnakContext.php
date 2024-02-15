<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\Tests\Fake;

use Wikibase\DataModel\Entity\Item;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\StatementListProvidingEntity;
use Wikibase\DataModel\Snak\Snak;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\AbstractContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursor;

/**
 * A constraint check context for a snak not connected to any statement.
 * This is a minimal Context implementation for tests.
 *
 * @license GPL-2.0-or-later
 */
class FakeSnakContext extends AbstractContext {

	/**
	 * @param Snak $snak
	 * @param StatementListProvidingEntity|null $entity defaults to a new Q1 item
	 */
	public function __construct(
		Snak $snak,
		StatementListProvidingEntity $entity = null
	) {
		parent::__construct(
			$entity ?: new Item( new ItemId( 'Q1' ) ),
			$snak
		);
	}

	public function getType(): string {
		return 'statement';
	}

	public function getSnakGroup( string $groupingMode, array $separators = [] ): array {
		return [ $this->snak ];
	}

	public function getCursor(): ContextCursor {
		return new AppendingContextCursor();
	}

}
