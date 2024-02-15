<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\Api;

use ApiBase;
use ApiMain;
use ApiResult;
use IBufferingStatsdDataFactory;
use InvalidArgumentException;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\Statement\StatementGuidParser;
use Wikibase\DataModel\Services\Statement\StatementGuidParsingException;
use Wikibase\Lib\LanguageFallbackChainFactory;
use Wikibase\Repo\Api\ApiErrorReporter;
use Wikibase\Repo\Api\ApiHelperFactory;
use WikibaseQuality\ConstraintReport\ConstraintCheck\DelegatingConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRendererFactory;
use Wikimedia\ParamValidator\ParamValidator;

/**
 * API module that checks whether the parameters of a constraint statement are valid.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CheckConstraintParameters extends ApiBase {

	public const PARAM_PROPERTY_ID = 'propertyid';
	public const PARAM_CONSTRAINT_ID = 'constraintid';
	public const KEY_STATUS = 'status';
	public const STATUS_OKAY = 'okay';
	public const STATUS_NOT_OKAY = 'not-okay';
	public const STATUS_NOT_FOUND = 'not-found';
	public const KEY_PROBLEMS = 'problems';
	public const KEY_MESSAGE_HTML = 'message-html';

	private ApiErrorReporter $apiErrorReporter;
	private LanguageFallbackChainFactory $languageFallbackChainFactory;
	private DelegatingConstraintChecker $delegatingConstraintChecker;
	private ViolationMessageRendererFactory $violationMessageRendererFactory;
	private StatementGuidParser $statementGuidParser;
	private IBufferingStatsdDataFactory $dataFactory;

	/**
	 * Creates new instance from global state.
	 */
	public static function newFromGlobalState(
		ApiMain $main,
		string $name,
		IBufferingStatsdDataFactory $dataFactory,
		ApiHelperFactory $apiHelperFactory,
		LanguageFallbackChainFactory $languageFallbackChainFactory,
		StatementGuidParser $statementGuidParser,
		DelegatingConstraintChecker $delegatingConstraintChecker,
		ViolationMessageRendererFactory $violationMessageRendererFactory
	): self {
		return new self(
			$main,
			$name,
			$apiHelperFactory,
			$languageFallbackChainFactory,
			$delegatingConstraintChecker,
			$violationMessageRendererFactory,
			$statementGuidParser,
			$dataFactory
		);
	}

	public function __construct(
		ApiMain $main,
		string $name,
		ApiHelperFactory $apiHelperFactory,
		LanguageFallbackChainFactory $languageFallbackChainFactory,
		DelegatingConstraintChecker $delegatingConstraintChecker,
		ViolationMessageRendererFactory $violationMessageRendererFactory,
		StatementGuidParser $statementGuidParser,
		IBufferingStatsdDataFactory $dataFactory
	) {
		parent::__construct( $main, $name );

		$this->apiErrorReporter = $apiHelperFactory->getErrorReporter( $this );
		$this->languageFallbackChainFactory = $languageFallbackChainFactory;
		$this->delegatingConstraintChecker = $delegatingConstraintChecker;
		$this->violationMessageRendererFactory = $violationMessageRendererFactory;
		$this->statementGuidParser = $statementGuidParser;
		$this->dataFactory = $dataFactory;
	}

	public function execute() {
		$this->dataFactory->increment(
			'wikibase.quality.constraints.api.checkConstraintParameters.execute'
		);

		$params = $this->extractRequestParams();
		$result = $this->getResult();

		$propertyIds = $this->parsePropertyIds( $params[self::PARAM_PROPERTY_ID] );
		$constraintIds = $this->parseConstraintIds( $params[self::PARAM_CONSTRAINT_ID] );

		$this->checkPropertyIds( $propertyIds, $result );
		$this->checkConstraintIds( $constraintIds, $result );

		$result->addValue( null, 'success', 1 );
	}

	/**
	 * @param array|null $propertyIdSerializations
	 * @return NumericPropertyId[]
	 */
	private function parsePropertyIds( ?array $propertyIdSerializations ): array {
		if ( $propertyIdSerializations === null ) {
			return [];
		} elseif ( empty( $propertyIdSerializations ) ) {
			$this->apiErrorReporter->dieError(
				'If ' . self::PARAM_PROPERTY_ID . ' is specified, it must be nonempty.',
				'no-data'
			);
		}

		return array_map(
			function ( $propertyIdSerialization ) {
				try {
					return new NumericPropertyId( $propertyIdSerialization );
				} catch ( InvalidArgumentException $e ) {
					$this->apiErrorReporter->dieError(
						"Invalid id: $propertyIdSerialization",
						'invalid-property-id',
						0, // default argument
						[ self::PARAM_PROPERTY_ID => $propertyIdSerialization ]
					);
				}
			},
			$propertyIdSerializations
		);
	}

	/**
	 * @param array|null $constraintIds
	 * @return string[]
	 */
	private function parseConstraintIds( ?array $constraintIds ): array {
		if ( $constraintIds === null ) {
			return [];
		} elseif ( empty( $constraintIds ) ) {
			$this->apiErrorReporter->dieError(
				'If ' . self::PARAM_CONSTRAINT_ID . ' is specified, it must be nonempty.',
				'no-data'
			);
		}

		return array_map(
			function ( $constraintId ) {
				try {
					$propertyId = $this->statementGuidParser->parse( $constraintId )->getEntityId();
					if ( !$propertyId instanceof NumericPropertyId ) {
						$this->apiErrorReporter->dieError(
							"Invalid property ID: {$propertyId->getSerialization()}",
							'invalid-property-id',
							0, // default argument
							[ self::PARAM_CONSTRAINT_ID => $constraintId ]
						);
					}
					return $constraintId;
				} catch ( StatementGuidParsingException $e ) {
					$this->apiErrorReporter->dieError(
						"Invalid statement GUID: $constraintId",
						'invalid-guid',
						0, // default argument
						[ self::PARAM_CONSTRAINT_ID => $constraintId ]
					);
				}
			},
			$constraintIds
		);
	}

	/**
	 * @param NumericPropertyId[] $propertyIds
	 * @param ApiResult $result
	 */
	private function checkPropertyIds( array $propertyIds, ApiResult $result ): void {
		foreach ( $propertyIds as $propertyId ) {
			$result->addArrayType( $this->getResultPathForPropertyId( $propertyId ), 'assoc' );
			$allConstraintExceptions = $this->delegatingConstraintChecker
				->checkConstraintParametersOnPropertyId( $propertyId );
			foreach ( $allConstraintExceptions as $constraintId => $constraintParameterExceptions ) {
				$this->addConstraintParameterExceptionsToResult(
					$constraintId,
					$constraintParameterExceptions,
					$result
				);
			}
		}
	}

	/**
	 * @param string[] $constraintIds
	 * @param ApiResult $result
	 */
	private function checkConstraintIds( array $constraintIds, ApiResult $result ): void {
		foreach ( $constraintIds as $constraintId ) {
			if ( $result->getResultData( $this->getResultPathForConstraintId( $constraintId ) ) ) {
				// already checked as part of checkPropertyIds()
				continue;
			}
			$constraintParameterExceptions = $this->delegatingConstraintChecker
				->checkConstraintParametersOnConstraintId( $constraintId );
			$this->addConstraintParameterExceptionsToResult( $constraintId, $constraintParameterExceptions, $result );
		}
	}

	/**
	 * @param NumericPropertyId $propertyId
	 * @return string[]
	 */
	private function getResultPathForPropertyId( NumericPropertyId $propertyId ): array {
		return [ $this->getModuleName(), $propertyId->getSerialization() ];
	}

	/**
	 * @param string $constraintId
	 * @return string[]
	 */
	private function getResultPathForConstraintId( string $constraintId ): array {
		$propertyId = $this->statementGuidParser->parse( $constraintId )->getEntityId();
		'@phan-var NumericPropertyId $propertyId';
		return array_merge( $this->getResultPathForPropertyId( $propertyId ), [ $constraintId ] );
	}

	/**
	 * Add the ConstraintParameterExceptions for $constraintId to the API result.
	 *
	 * @param string $constraintId
	 * @param ConstraintParameterException[]|null $constraintParameterExceptions
	 * @param ApiResult $result
	 */
	private function addConstraintParameterExceptionsToResult(
		string $constraintId,
		?array $constraintParameterExceptions,
		ApiResult $result
	): void {
		$path = $this->getResultPathForConstraintId( $constraintId );
		if ( $constraintParameterExceptions === null ) {
			$result->addValue(
				$path,
				self::KEY_STATUS,
				self::STATUS_NOT_FOUND
			);
		} else {
			$result->addValue(
				$path,
				self::KEY_STATUS,
				empty( $constraintParameterExceptions ) ? self::STATUS_OKAY : self::STATUS_NOT_OKAY
			);

			$language = $this->getLanguage();
			$violationMessageRenderer = $this->violationMessageRendererFactory
				->getViolationMessageRenderer(
					$language,
					$this->languageFallbackChainFactory->newFromLanguage( $language ),
					$this
				);
			$problems = [];
			foreach ( $constraintParameterExceptions as $constraintParameterException ) {
				$problems[] = [
					self::KEY_MESSAGE_HTML => $violationMessageRenderer->render(
						$constraintParameterException->getViolationMessage() ),
				];
			}
			$result->addValue(
				$path,
				self::KEY_PROBLEMS,
				$problems
			);
		}
	}

	/**
	 * @return array[]
	 * @codeCoverageIgnore
	 */
	public function getAllowedParams() {
		return [
			self::PARAM_PROPERTY_ID => [
				ParamValidator::PARAM_TYPE => 'string',
				ParamValidator::PARAM_ISMULTI => true,
			],
			self::PARAM_CONSTRAINT_ID => [
				ParamValidator::PARAM_TYPE => 'string',
				ParamValidator::PARAM_ISMULTI => true,
			],
		];
	}

	/**
	 * @return string[]
	 * @codeCoverageIgnore
	 */
	public function getExamplesMessages() {
		return [
			'action=wbcheckconstraintparameters&propertyid=P247'
				=> 'apihelp-wbcheckconstraintparameters-example-propertyid-1',
			'action=wbcheckconstraintparameters&' .
			'constraintid=P247$0fe1711e-4c0f-82ce-3af0-830b721d0fba|' .
			'P225$cdc71e4a-47a0-12c5-dfb3-3f6fc0b6613f'
				=> 'apihelp-wbcheckconstraintparameters-example-constraintid-2',
		];
	}

}
