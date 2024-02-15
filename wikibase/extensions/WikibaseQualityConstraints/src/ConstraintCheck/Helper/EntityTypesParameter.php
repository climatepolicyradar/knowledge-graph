<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use Wikibase\DataModel\Entity\ItemId;

/**
 * Wrapper class for a constraint parameter representing a list of entity types.
 *
 * @license GPL-2.0-or-later
 * @author Lucas Werkmeister
 */
class EntityTypesParameter {

	/**
	 * @var string[]
	 */
	private $entityTypes;

	/**
	 * @var ItemId[]
	 */
	private $entityTypeItemIds;

	/**
	 * @param string[] $entityTypes The allowed entity types as strings.
	 * @param ItemId[] $entityTypeItemIds The item IDs of the allowed entity types.
	 */
	public function __construct(
		array $entityTypes,
		array $entityTypeItemIds
	) {
		$this->entityTypes = $entityTypes;
		$this->entityTypeItemIds = $entityTypeItemIds;
	}

	/**
	 * @return string[] The allowed entity types as strings.
	 * @see EntityDocument::getType()
	 */
	public function getEntityTypes() {
		return $this->entityTypes;
	}

	/**
	 * @return ItemId[] The item IDs of the allowed entity types.
	 */
	public function getEntityTypeItemIds() {
		return $this->entityTypeItemIds;
	}

}
