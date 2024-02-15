<?php

namespace WikibaseQuality\ConstraintReport\Maintenance;

use Config;
use Deserializers\Deserializer;
use Maintenance;
use MediaWiki\Http\HttpRequestFactory;
use MediaWiki\MediaWikiServices;
use MediaWiki\WikiMap\WikiMap;
use Serializers\Serializer;
use User;
use Wikibase\DataModel\Entity\Item;
use Wikibase\DataModel\SiteLinkList;
use Wikibase\DataModel\Statement\StatementListProvider;
use Wikibase\Lib\Store\EntityStore;
use Wikibase\Lib\Store\StorageException;
use Wikibase\Repo\WikibaseRepo;

// @codeCoverageIgnoreStart
$basePath = getenv( "MW_INSTALL_PATH" ) !== false
	? getenv( "MW_INSTALL_PATH" ) : __DIR__ . "/../../..";

require_once $basePath . "/maintenance/Maintenance.php";
// @codeCoverageIgnoreEnd

/**
 * Imports entities needed for constraint checks from Wikidata into the local repository.
 *
 * @license GPL-2.0-or-later
 */
class ImportConstraintEntities extends Maintenance {

	/**
	 * @var Serializer
	 */
	private $entitySerializer;

	/**
	 * @var Deserializer
	 */
	private $entityDeserializer;

	/**
	 * @var EntityStore
	 */
	private $entityStore;

	/**
	 * @var HttpRequestFactory
	 */
	private $httpRequestFactory;

	/**
	 * @var User|null (null in dry-run mode, non-null otherwise)
	 */
	private $user;

	public function __construct() {
		parent::__construct();

		$this->addDescription(
			'Import entities needed for constraint checks ' .
			'from Wikidata into the local repository.'
		);
		$this->addOption(
			'config-format',
			'The format in which the resulting configuration will be omitted: ' .
			'"globals" for directly settings global variables, suitable for inclusion in LocalSettings.php (default), ' .
			'or "wgConf" for printing parts of arrays suitable for inclusion in $wgConf->settings.'
		);
		$this->addOption(
			'dry-run',
			'Don’t actually import entities, just print which ones would be imported.'
		);
		$this->requireExtension( 'WikibaseQualityConstraints' );
	}

	/**
	 * (This cannot happen in the constructor because the autoloader is not yet initialized there.)
	 */
	private function setupServices() {
		$services = MediaWikiServices::getInstance();
		$this->entitySerializer = WikibaseRepo::getAllTypesEntitySerializer( $services );
		$this->entityDeserializer = WikibaseRepo::getInternalFormatEntityDeserializer( $services );
		$this->entityStore = WikibaseRepo::getEntityStore( $services );
		$this->httpRequestFactory = $services->getHttpRequestFactory();
		if ( !$this->getOption( 'dry-run', false ) ) {
			$this->user = User::newSystemUser( 'WikibaseQualityConstraints importer' );
		}
	}

	public function execute() {
		$this->setupServices();

		$configUpdates = [];

		$extensionJsonFile = __DIR__ . '/../extension.json';
		$extensionJsonText = file_get_contents( $extensionJsonFile );
		$extensionJson = json_decode( $extensionJsonText, /* assoc = */ true );
		// @phan-suppress-next-line PhanTypeArraySuspiciousNullable
		$wikidataEntityIds = $this->getEntitiesToImport( $extensionJson['config'], $this->getConfig() );

		foreach ( $wikidataEntityIds as $key => $wikidataEntityId ) {
			$localEntityId = $this->importEntityFromWikidata( $wikidataEntityId );
			$configUpdates[$key] = [
				'wikidata' => $wikidataEntityId,
				'local' => $localEntityId,
			];
		}

		$this->outputConfigUpdates( $configUpdates );
	}

	/**
	 * @param array[] $extensionJsonConfig
	 * @param Config $wikiConfig
	 * @return string[]
	 */
	private function getEntitiesToImport( array $extensionJsonConfig, Config $wikiConfig ) {
		$wikidataEntityIds = [];

		foreach ( $extensionJsonConfig as $key => $value ) {
			if ( !preg_match( '/Id$/', $key ) ) {
				continue;
			}

			$wikidataEntityId = $value['value'];
			$localEntityId = $wikiConfig->get( $key );

			if ( $localEntityId === $wikidataEntityId ) {
				$wikidataEntityIds[$key] = $wikidataEntityId;
			}
		}

		return $wikidataEntityIds;
	}

	/**
	 * @param string $wikidataEntityId
	 * @return string local entity ID
	 */
	private function importEntityFromWikidata( $wikidataEntityId ) {
		$wikidataEntityUrl = "https://www.wikidata.org/wiki/Special:EntityData/$wikidataEntityId.json";
		$wikidataEntitiesJson = $this->httpRequestFactory->get( $wikidataEntityUrl, [], __METHOD__ );
		return $this->importEntityFromJson( $wikidataEntityId, $wikidataEntitiesJson );
	}

	/**
	 * @param string $wikidataEntityId
	 * @param string $wikidataEntitiesJson
	 * @return string local entity ID
	 */
	private function importEntityFromJson( $wikidataEntityId, $wikidataEntitiesJson ) {
		// @phan-suppress-next-line PhanTypeArraySuspiciousNullable
		$wikidataEntityArray = json_decode( $wikidataEntitiesJson, true )['entities'][$wikidataEntityId];
		$wikidataEntity = $this->entityDeserializer->deserialize( $wikidataEntityArray );

		$wikidataEntity->setId( null );

		if ( $wikidataEntity instanceof StatementListProvider ) {
			$wikidataEntity->getStatements()->clear();
		}

		if ( $wikidataEntity instanceof Item ) {
			$wikidataEntity->setSiteLinkList( new SiteLinkList() );
		}

		if ( $this->getOption( 'dry-run', false ) ) {
			$wikidataEntityJson = json_encode( $this->entitySerializer->serialize( $wikidataEntity ) );
			$this->output( $wikidataEntityJson . "\n" );
			return "-$wikidataEntityId";
		}

		try {
			$localEntity = $this->entityStore->saveEntity(
				$wikidataEntity,
				"imported from [[wikidata:$wikidataEntityId]]",
				$this->user,
				EDIT_NEW | EDIT_FORCE_BOT
			)->getEntity();

			return $localEntity->getId()->getSerialization();
		} catch ( StorageException $storageException ) {
			return $this->storageExceptionToEntityId( $storageException );
		}
	}

	private function storageExceptionToEntityId( StorageException $storageException ) {
		$message = $storageException->getMessage();
		// example messages:
		// * Item [[Item:Q475|Q475]] already has label "as references"
		//   associated with language code en, using the same description text.
		// * Item [[Q475]] already has label "as references"
		//   associated with language code en, using the same description text.
		// * Property [[Property:P694|P694]] already has label "instance of"
		//   associated with language code en.
		$pattern = '/[[|]([^][|]*)]] already has label .* associated with language code/';
		if ( preg_match( $pattern, $message, $matches ) ) {
			return $matches[1];
		} else {
			throw $storageException;
		}
	}

	private function outputConfigUpdates( array $configUpdates ) {
		$configFormat = $this->getOption( 'config-format', 'globals' );
		switch ( $configFormat ) {
			case 'globals':
				$this->outputConfigUpdatesGlobals( $configUpdates );
				break;
			case 'wgConf':
				$this->outputConfigUpdatesWgConf( $configUpdates );
				break;
			default:
				$this->error( "Invalid config format \"$configFormat\", using \"globals\"" );
				$this->outputConfigUpdatesGlobals( $configUpdates );
				break;
		}
	}

	/**
	 * @param array[] $configUpdates
	 */
	private function outputConfigUpdatesGlobals( array $configUpdates ) {
		foreach ( $configUpdates as $key => $value ) {
			$localValueCode = var_export( $value['local'], true );
			$this->output( "\$wg$key = $localValueCode;\n" );
		}
	}

	/**
	 * @param array[] $configUpdates
	 */
	private function outputConfigUpdatesWgConf( array $configUpdates ) {
		$wikiIdCode = var_export( WikiMap::getCurrentWikiId(), true );
		foreach ( $configUpdates as $key => $value ) {
			$keyCode = var_export( "wg$key", true );
			$wikidataValueCode = var_export( $value['wikidata'], true );
			$localValueCode = var_export( $value['local'], true );
			$block = <<< EOF
$keyCode => [
	'default' => $wikidataValueCode,
	$wikiIdCode => $localValueCode,
],


EOF;
			$this->output( $block );
		}
	}

}

// @codeCoverageIgnoreStart
$maintClass = ImportConstraintEntities::class;
require_once RUN_MAINTENANCE_IF_MAIN;
// @codeCoverageIgnoreEnd
