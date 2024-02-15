<?php

namespace WikibaseQuality\ConstraintReport;

use Config;
use ExtensionRegistry;
use JobSpecification;
use MediaWiki\Hook\BeforePageDisplayHook;
use MediaWiki\MediaWikiServices;
use MediaWiki\Page\Hook\ArticlePurgeHook;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\Lib\Changes\Change;
use Wikibase\Lib\Changes\EntityChange;
use Wikibase\Lib\Changes\EntityDiffChangedAspects;
use Wikibase\Repo\WikibaseRepo;
use WikibaseQuality\ConstraintReport\Api\ResultsCache;
use WikibaseQuality\ConstraintReport\Job\CheckConstraintsJob;

/**
 * Container for hook callbacks registered in extension.json.
 *
 * @license GPL-2.0-or-later
 */
final class WikibaseQualityConstraintsHooks implements
	ArticlePurgeHook,
	BeforePageDisplayHook
{

	public static function onWikibaseChange( Change $change ) {
		if ( !( $change instanceof EntityChange ) ) {
			return;
		}
		/** @var EntityChange $change */

		$services = MediaWikiServices::getInstance();
		$config = $services->getMainConfig();
		$jobQueueGroup = $services->getJobQueueGroup();

		// If jobs are enabled and the results would be stored in some way run a job.
		if (
			$config->get( 'WBQualityConstraintsEnableConstraintsCheckJobs' ) &&
			$config->get( 'WBQualityConstraintsCacheCheckConstraintsResults' ) &&
			self::isSelectedForJobRunBasedOnPercentage()
		) {
			$params = [ 'entityId' => $change->getEntityId()->getSerialization() ];
			$jobQueueGroup->lazyPush(
				new JobSpecification( CheckConstraintsJob::COMMAND, $params )
			);
		}

		if ( $config->get( 'WBQualityConstraintsEnableConstraintsImportFromStatements' ) &&
			self::isConstraintStatementsChange( $config, $change )
		) {
			$params = [ 'propertyId' => $change->getEntityId()->getSerialization() ];
			$metadata = $change->getMetadata();
			if ( array_key_exists( 'rev_id', $metadata ) ) {
				$params['revisionId'] = $metadata['rev_id'];
			}
			$jobQueueGroup->push(
				new JobSpecification( 'constraintsTableUpdate', $params )
			);
		}
	}

	private static function isSelectedForJobRunBasedOnPercentage() {
		$config = MediaWikiServices::getInstance()->getMainConfig();
		$percentage = $config->get( 'WBQualityConstraintsEnableConstraintsCheckJobsRatio' );

		return mt_rand( 1, 100 ) <= $percentage;
	}

	public static function isConstraintStatementsChange( Config $config, Change $change ) {
		if ( !( $change instanceof EntityChange ) ||
			 $change->getAction() !== EntityChange::UPDATE ||
			 !( $change->getEntityId() instanceof NumericPropertyId )
		) {
			return false;
		}

		$info = $change->getInfo();

		if ( !array_key_exists( 'compactDiff', $info ) ) {
			// the non-compact diff ($info['diff']) does not contain statement diffs (T110996),
			// so we only know that the change *might* affect the constraint statements
			return true;
		}

		/** @var EntityDiffChangedAspects $aspects */
		$aspects = $info['compactDiff'];

		$propertyConstraintId = $config->get( 'WBQualityConstraintsPropertyConstraintId' );
		return in_array( $propertyConstraintId, $aspects->getStatementChanges() );
	}

	public function onArticlePurge( $wikiPage ) {
		$entityContentFactory = WikibaseRepo::getEntityContentFactory();
		if ( $entityContentFactory->isEntityContentModel( $wikiPage->getContentModel() ) ) {
			$entityIdLookup = WikibaseRepo::getEntityIdLookup();
			$entityId = $entityIdLookup->getEntityIdForTitle( $wikiPage->getTitle() );
			if ( $entityId !== null ) {
				$resultsCache = ResultsCache::getDefaultInstance();
				$resultsCache->delete( $entityId );
			}
		}
	}

	public function onBeforePageDisplay( $out, $skin ): void {
		$lookup = WikibaseRepo::getEntityNamespaceLookup();
		$title = $out->getTitle();
		if ( $title === null ) {
			return;
		}

		if ( !$lookup->isNamespaceWithEntities( $title->getNamespace() ) ) {
			return;
		}
		if ( empty( $out->getJsConfigVars()['wbIsEditView'] ) ) {
			return;
		}

		$services = MediaWikiServices::getInstance();
		$config = $services->getMainConfig();

		$isMobileView = ExtensionRegistry::getInstance()->isLoaded( 'MobileFrontend' ) &&
			$services->getService( 'MobileFrontend.Context' )->shouldDisplayMobileView();
		if ( $isMobileView ) {
			return;
		}

		$out->addModules( 'wikibase.quality.constraints.suggestions' );

		if ( $config->get( 'WBQualityConstraintsShowConstraintViolationToNonLoggedInUsers' )
			|| $out->getUser()->isRegistered() ) {
				$out->addModules( 'wikibase.quality.constraints.gadget' );
		}
	}

}
