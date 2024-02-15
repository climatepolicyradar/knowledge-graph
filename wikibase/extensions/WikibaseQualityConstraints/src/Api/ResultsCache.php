<?php

namespace WikibaseQuality\ConstraintReport\Api;

use MediaWiki\MediaWikiServices;
use WANObjectCache;
use Wikibase\DataModel\Entity\EntityId;

/**
 * A thin wrapper around a WANObjectCache
 * that maps entity IDs to cache keys.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ResultsCache {

	/**
	 * @var WANObjectCache
	 */
	private $cache;

	/**
	 * @var string
	 */
	private $formatVersion;

	public static function getDefaultInstance() {
		return new self(
			MediaWikiServices::getInstance()->getMainWANObjectCache(),
			'v2.2' // .1: T188384; .2: T189593
		);
	}

	/**
	 * @param WANObjectCache $cache
	 * @param string $formatVersion The version of the API response format.
	 */
	public function __construct( WANObjectCache $cache, $formatVersion ) {
		$this->cache = $cache;
		$this->formatVersion = $formatVersion;
	}

	/**
	 * @param EntityId $entityId
	 * @return string cache key
	 */
	public function makeKey( EntityId $entityId ) {
		return $this->cache->makeKey(
			'WikibaseQualityConstraints', // extension
			'checkConstraints', // action
			$this->formatVersion, // API response format version
			$entityId->getSerialization()
		);
	}

	/**
	 * @param EntityId $key
	 * @param mixed &$curTTL
	 * @param string[] $checkKeys
	 * @param array &$info
	 * @return mixed
	 */
	public function get( EntityId $key, &$curTTL = null, array $checkKeys = [], array &$info = [] ) {
		return $this->cache->get( $this->makeKey( $key ), $curTTL, $checkKeys, $info );
	}

	/**
	 * @param EntityId $key
	 * @param mixed $value
	 * @param int $ttl
	 * @param array $opts
	 * @return bool
	 */
	public function set( EntityId $key, $value, $ttl = 0, array $opts = [] ) {
		return $this->cache->set( $this->makeKey( $key ), $value, $ttl, $opts );
	}

	/**
	 * @param EntityId $key
	 * @return bool
	 */
	public function delete( EntityId $key ) {
		return $this->cache->delete( $this->makeKey( $key ) );
	}

}
