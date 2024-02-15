<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Cache;

/**
 * A bool along with information whether and how it was cached.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CachedBool {

	/**
	 * @var bool
	 */
	private $bool;

	/**
	 * @var Metadata
	 */
	private $metadata;

	/**
	 * @param bool $bool
	 * @param Metadata $metadata
	 */
	public function __construct( $bool, Metadata $metadata ) {
		$this->bool = $bool;
		$this->metadata = $metadata;
	}

	/**
	 * @return bool
	 */
	public function getBool() {
		return $this->bool;
	}

	/**
	 * @return Metadata
	 */
	public function getMetadata() {
		return $this->metadata;
	}

}
