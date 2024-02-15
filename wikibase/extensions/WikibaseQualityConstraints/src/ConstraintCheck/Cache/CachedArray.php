<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Cache;

/**
 * An array (of unspecified nature) along with information whether and how it was cached.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CachedArray {

	/**
	 * @var array
	 */
	private $array;

	/**
	 * @var Metadata
	 */
	private $metadata;

	public function __construct( array $array, Metadata $metadata ) {
		$this->array = $array;
		$this->metadata = $metadata;
	}

	/**
	 * @return array
	 */
	public function getArray() {
		return $this->array;
	}

	/**
	 * @return Metadata
	 */
	public function getMetadata() {
		return $this->metadata;
	}

}
