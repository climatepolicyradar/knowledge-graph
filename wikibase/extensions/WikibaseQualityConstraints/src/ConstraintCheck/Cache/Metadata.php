<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Cache;

use Wikimedia\Assert\Assert;

/**
 * Collection of information about a value.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class Metadata {

	/**
	 * @var CachingMetadata
	 */
	private $cachingMetadata;

	/**
	 * @var DependencyMetadata
	 */
	private $dependencyMetadata;

	/**
	 * @return self Empty collection.
	 */
	public static function blank() {
		$ret = new self;
		$ret->cachingMetadata = CachingMetadata::fresh();
		$ret->dependencyMetadata = DependencyMetadata::blank();
		return $ret;
	}

	public static function ofCachingMetadata( CachingMetadata $cachingMetadata ) {
		$ret = new self;
		$ret->cachingMetadata = $cachingMetadata;
		$ret->dependencyMetadata = DependencyMetadata::blank();
		return $ret;
	}

	public static function ofDependencyMetadata( DependencyMetadata $dependencyMetadata ) {
		$ret = new self;
		$ret->cachingMetadata = CachingMetadata::fresh();
		$ret->dependencyMetadata = $dependencyMetadata;
		return $ret;
	}

	/**
	 * @param self[] $metadatas
	 * @return self
	 */
	public static function merge( array $metadatas ) {
		Assert::parameterElementType( self::class, $metadatas, '$metadatas' );
		$cachingMetadatas = [];
		$dependencyMetadatas = [];
		foreach ( $metadatas as $metadata ) {
			$cachingMetadatas[] = $metadata->cachingMetadata;
			$dependencyMetadatas[] = $metadata->dependencyMetadata;
		}
		$ret = new self;
		$ret->cachingMetadata = CachingMetadata::merge( $cachingMetadatas );
		$ret->dependencyMetadata = DependencyMetadata::merge( $dependencyMetadatas );
		return $ret;
	}

	/**
	 * @return CachingMetadata
	 */
	public function getCachingMetadata() {
		return $this->cachingMetadata;
	}

	/**
	 * @return DependencyMetadata
	 */
	public function getDependencyMetadata() {
		return $this->dependencyMetadata;
	}

}
