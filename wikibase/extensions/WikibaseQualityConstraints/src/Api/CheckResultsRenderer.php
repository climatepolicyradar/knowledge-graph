<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\Api;

use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\EntityId\EntityIdFormatter;
use Wikibase\Lib\Store\EntityTitleLookup;
use Wikibase\Lib\TermLanguageFallbackChain;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckConstraintsResponse;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckResults;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\NullResult;

/**
 * Converts check results into arrays and stores them inside one big response array.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CheckResultsRenderer {

	private EntityTitleLookup $entityTitleLookup;
	private EntityIdFormatter $entityIdLabelFormatter;
	private TermLanguageFallbackChain $languageFallbackChain;
	private ViolationMessageRenderer $violationMessageRenderer;

	public function __construct(
		EntityTitleLookup $entityTitleLookup,
		EntityIdFormatter $entityIdLabelFormatter,
		TermLanguageFallbackChain $languageFallbackChain,
		ViolationMessageRenderer $violationMessageRenderer
	) {
		$this->entityTitleLookup = $entityTitleLookup;
		$this->entityIdLabelFormatter = $entityIdLabelFormatter;
		$this->languageFallbackChain = $languageFallbackChain;
		$this->violationMessageRenderer = $violationMessageRenderer;
	}

	public function render( CachedCheckResults $checkResults ): CachedCheckConstraintsResponse {
		$response = [];
		foreach ( $checkResults->getArray() as $checkResult ) {
			$resultArray = $this->checkResultToArray( $checkResult );
			$checkResult->getContextCursor()->storeCheckResultInArray( $resultArray, $response );
		}
		return new CachedCheckConstraintsResponse(
			$response,
			$checkResults->getMetadata()
		);
	}

	public function checkResultToArray( CheckResult $checkResult ): ?array {
		if ( $checkResult instanceof NullResult ) {
			return null;
		}

		$constraintId = $checkResult->getConstraint()->getConstraintId();
		$typeItemId = $checkResult->getConstraint()->getConstraintTypeItemId();
		$constraintPropertyId = new NumericPropertyId( $checkResult->getContextCursor()->getSnakPropertyId() );

		$title = $this->entityTitleLookup->getTitleForId( $constraintPropertyId );
		$talkTitle = $title->getTalkPageIfDefined();
		$typeLabel = $this->entityIdLabelFormatter->formatEntityId( new ItemId( $typeItemId ) );
		$link = $title->getFullURL() . '#' . $constraintId;

		$constraint = [
			'id' => $constraintId,
			'type' => $typeItemId,
			'typeLabel' => $typeLabel,
			'link' => $link,
			'discussLink' => $talkTitle ? $talkTitle->getFullURL() : $title->getFullURL(),
		];

		$result = [
			'status' => $checkResult->getStatus(),
			'property' => $constraintPropertyId->getSerialization(),
			'constraint' => $constraint,
		];
		$message = $checkResult->getMessage();
		if ( $message ) {
			$result['message-html'] = $this->violationMessageRenderer->render( $message );
		}
		$constraintClarification = $this->renderConstraintClarification( $checkResult );
		if ( $constraintClarification !== null ) {
			$result['constraint-clarification'] = $constraintClarification;
		}
		if ( $checkResult->getContextCursor()->getType() === Context::TYPE_STATEMENT ) {
			$result['claim'] = $checkResult->getContextCursor()->getStatementGuid();
		}
		$cachingMetadataArray = $checkResult->getMetadata()->getCachingMetadata()->toArray();
		if ( $cachingMetadataArray !== null ) {
			$result['cached'] = $cachingMetadataArray;
		}

		return $result;
	}

	private function renderConstraintClarification( CheckResult $result ): ?string {
		$texts = $result->getConstraintClarification()->getTexts();
		foreach ( $this->languageFallbackChain->getFetchLanguageCodes() as $languageCode ) {
			if ( array_key_exists( $languageCode, $texts ) ) {
				return $texts[$languageCode]->getText();
			}
		}

		return null;
	}

}
