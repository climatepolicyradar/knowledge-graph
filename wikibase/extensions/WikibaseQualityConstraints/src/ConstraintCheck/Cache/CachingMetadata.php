<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Cache;

use Wikimedia\Assert\Assert;

/**
 * Information about whether and how a value was cached.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CachingMetadata {

	/**
	 * @var int|bool The maximum age in seconds,
	 * or false to indicate that the value wasn’t cached.
	 */
	private $maxAge = false;

	/**
	 * @return self Indication that a value is fresh, i. e. not cached.
	 */
	public static function fresh() {
		return new self;
	}

	/**
	 * @param int $maxAge The maximum age of the cached value (in seconds).
	 * @return self Indication that a value is possibly outdated by up to this many seconds.
	 */
	public static function ofMaximumAgeInSeconds( $maxAge ) {
		Assert::parameterType( 'integer', $maxAge, '$maxAge' );
		Assert::parameter( $maxAge > 0, '$maxAge', '$maxage > 0' );
		$ret = new self;
		$ret->maxAge = $maxAge;
		return $ret;
	}

	/**
	 * Deserializes the metadata from an array (or null if the value is fresh).
	 * @param array|null $array As returned by toArray.
	 * @return self
	 */
	public static function ofArray( array $array = null ) {
		$ret = new self;
		if ( $array !== null ) {
			$ret->maxAge = $array['maximumAgeInSeconds'];
		}
		return $ret;
	}

	/**
	 * @param self[] $metadatas
	 * @return self
	 */
	public static function merge( array $metadatas ) {
		Assert::parameterElementType( self::class, $metadatas, '$metadatas' );
		$ret = new self;
		foreach ( $metadatas as $metadata ) {
			$ret->maxAge = max( $ret->maxAge, $metadata->maxAge );
		}
		return $ret;
	}

	/**
	 * @return bool Whether the value is cached or not (fresh).
	 */
	public function isCached() {
		return $this->maxAge !== false;
	}

	/**
	 * @return int The maximum age of the cached value (in seconds), in other words:
	 * the value might be outdated by up to this many seconds.
	 * For a fresh value, returns 0.
	 */
	public function getMaximumAgeInSeconds() {
		if ( is_int( $this->maxAge ) ) {
			return $this->maxAge;
		} else {
			return 0;
		}
	}

	/**
	 * Serializes the metadata into an array (or null if the value is fresh).
	 * @return array|null
	 */
	public function toArray() {
		return $this->isCached() ?
			[
				'maximumAgeInSeconds' => $this->maxAge,
			] :
			null;
	}

}
