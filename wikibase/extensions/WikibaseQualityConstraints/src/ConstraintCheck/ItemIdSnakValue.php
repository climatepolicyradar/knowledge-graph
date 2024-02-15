<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck;

use DomainException;
use InvalidArgumentException;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Snak\Snak;

/**
 * A value that can either be an item ID, some value (unknown value), or no value.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ItemIdSnakValue {

	/**
	 * @var ItemId|null
	 */
	private $itemId = null;

	/**
	 * @var bool
	 */
	private $some = false;

	/**
	 * @var bool
	 */
	private $no = false;

	private function __construct() {
	}

	/**
	 * Get an {@link ItemIdSnakValue} from the given $itemId.
	 *
	 * @param ItemId $itemId
	 * @return self
	 */
	public static function fromItemId( ItemId $itemId ) {
		$ret = new self;
		$ret->itemId = $itemId;
		return $ret;
	}

	/**
	 * Get an {@link ItemIdSnakValue} that wraps an unknown value.
	 *
	 * @return self
	 */
	public static function someValue() {
		$ret = new self;
		$ret->some = true;
		return $ret;
	}

	/**
	 * Get an {@link ItemIdSnakValue} that wraps no value.
	 *
	 * @return self
	 */
	public static function noValue() {
		$ret = new self;
		$ret->no = true;
		return $ret;
	}

	/**
	 * Get an {@link ItemIdSnakValue} that matches the given snak.
	 *
	 * @param Snak $snak
	 *
	 * @throws InvalidArgumentException
	 * @return self
	 */
	public static function fromSnak( Snak $snak ) {
		switch ( true ) {
			case $snak instanceof PropertyValueSnak:
				$dataValue = $snak->getDataValue();
				if ( $dataValue instanceof EntityIdValue ) {
					$itemId = $dataValue->getEntityId();
					if ( $itemId instanceof ItemId ) {
						return self::fromItemId( $itemId );
					}
				}
				break;
			case $snak instanceof PropertySomeValueSnak:
				return self::someValue();
			case $snak instanceof PropertyNoValueSnak:
				return self::noValue();
		}

		throw new InvalidArgumentException( 'Snak must contain item ID value or be a somevalue / novalue snak' );
	}

	/**
	 * Check whether this {@link ItemIdSnakValue} contains a known value or not.
	 *
	 * @return bool
	 */
	public function isValue() {
		return $this->itemId !== null;
	}

	/**
	 * Check whether this {@link ItemIdSnakValue} contains an unknown value or not.
	 *
	 * @return bool
	 */
	public function isSomeValue() {
		return $this->some;
	}

	/**
	 * Check whether this {@link ItemIdSnakValue} contains no value or not.
	 *
	 * @return bool
	 */
	public function isNoValue() {
		return $this->no;
	}

	/**
	 * Get the item ID contained in this {@link ItemIdSnakValue}.
	 * Only valid if {@link isValue} is true.
	 *
	 * @throws DomainException if this value does not contain an item ID
	 * @return ItemId
	 */
	public function getItemId() {
		if ( !$this->isValue() ) {
			throw new DomainException( 'This value does not contain an item ID.' );
		}
		return $this->itemId;
	}

	/**
	 * Check whether this value matches the given $snak
	 * (same kind and, if contains known value, same value).
	 *
	 * @param Snak $snak
	 * @return bool
	 */
	public function matchesSnak( Snak $snak ) {
		switch ( true ) {
			case $snak instanceof PropertyValueSnak:
				$dataValue = $snak->getDataValue();
				return $this->isValue() &&
					$dataValue instanceof EntityIdValue &&
					$dataValue->getEntityId() instanceof ItemId &&
					$dataValue->getEntityId()->equals( $this->getItemId() );
			case $snak instanceof PropertySomeValueSnak:
				return $this->isSomeValue();
			case $snak instanceof PropertyNoValueSnak:
				return $this->isNoValue();
		}
	}

}
