<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

use MediaWiki\Site\MediaWikiPageNameNormalizer;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Services\Lookup\PropertyDataTypeLookup;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class CommonsLinkChecker implements ConstraintChecker {

	/**
	 * @var ConstraintParameterParser
	 */
	private $constraintParameterParser;

	/**
	 * @var MediaWikiPageNameNormalizer
	 */
	private $pageNameNormalizer;

	/**
	 * @var PropertyDataTypeLookup
	 */
	private $propertyDatatypeLookup;

	public function __construct(
		ConstraintParameterParser $constraintParameterParser,
		MediaWikiPageNameNormalizer $pageNameNormalizer,
		PropertyDataTypeLookup $propertyDatatypeLookup
	) {
		$this->constraintParameterParser = $constraintParameterParser;
		$this->pageNameNormalizer = $pageNameNormalizer;
		$this->propertyDatatypeLookup = $propertyDatatypeLookup;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getSupportedContextTypes(): array {
		return self::ALL_CONTEXT_TYPES_SUPPORTED;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getDefaultContextTypes(): array {
		return Context::ALL_CONTEXT_TYPES;
	}

	/** @codeCoverageIgnore This method is purely declarative. */
	public function getSupportedEntityTypes() {
		return self::ALL_ENTITY_TYPES_SUPPORTED;
	}

	/**
	 * Get the number of a namespace on Wikimedia Commons (commonswiki).
	 * All namespaces not known to this function will be looked up by the TitleParser.
	 *
	 * @return array first element is the namespace number (default namespace for TitleParser),
	 * second element is a string to prepend to the title before giving it to the TitleParser
	 */
	private function getCommonsNamespace( string $namespace ): array {
		switch ( $namespace ) {
			case '':
				return [ NS_MAIN, '' ];
			// extra namespaces, see operations/mediawiki-config.git,
			// wmf-config/InitialiseSettings.php, 'wgExtraNamespaces' key, 'commonswiki' subkey
			case 'Creator':
				return [ 100, '' ];
			case 'TimedText':
				return [ 102, '' ];
			case 'Sequence':
				return [ 104, '' ];
			case 'Institution':
				return [ 106, '' ];
			// extension namespace, see mediawiki/extensions/JsonConfig.git,
			// extension.json, 'namespaces' key, third element
			case 'Data':
				return [ 486, '' ];
			default:
				return [ NS_MAIN, $namespace . ':' ];
		}
	}

	/**
	 * Checks 'Commons link' constraint.
	 *
	 * @throws ConstraintParameterException
	 */
	public function checkConstraint( Context $context, Constraint $constraint ): CheckResult {
		$constraintParameters = $constraint->getConstraintParameters();
		$constraintTypeItemId = $constraint->getConstraintTypeItemId();

		$namespace = $this->constraintParameterParser->parseNamespaceParameter(
			$constraintParameters,
			$constraintTypeItemId
		);

		$snak = $context->getSnak();

		if ( !$snak instanceof PropertyValueSnak ) {
			// nothing to check
			return new CheckResult( $context, $constraint, CheckResult::STATUS_COMPLIANCE );
		}

		$dataValue = $snak->getDataValue();

		/*
		 * error handling:
		 *   type of $dataValue for properties with 'Commons link' constraint has to be 'string'
		 *   parameter $namespace can be null, works for commons galleries
		 */
		if ( $dataValue->getType() !== 'string' ) {
			$message = ( new ViolationMessage( 'wbqc-violation-message-value-needed-of-type' ) )
				->withEntityId( new ItemId( $constraintTypeItemId ), Role::CONSTRAINT_TYPE_ITEM )
				->withDataValueType( 'string' );
			return new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION, $message );
		}

		$commonsLink = $dataValue->getValue();
		if ( !$this->commonsLinkIsWellFormed( $commonsLink ) ) {
			return new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION,
				new ViolationMessage( 'wbqc-violation-message-commons-link-not-well-formed' ) );
		}

		$dataType = $this->propertyDatatypeLookup->getDataTypeIdForProperty( $snak->getPropertyId() );
		switch ( $dataType ) {
			case 'geo-shape':
			case 'tabular-data':
				if ( strpos( $commonsLink, $namespace . ':' ) !== 0 ) {
					return new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION,
						new ViolationMessage( 'wbqc-violation-message-commons-link-not-well-formed' ) );
				}
				$pageName = $commonsLink;
				break;
			default:
				$pageName = $namespace ? $namespace . ':' . $commonsLink : $commonsLink;
				break;
		}

		$prefix = $this->getCommonsNamespace( $namespace )[1];
		$normalizedTitle = $this->pageNameNormalizer->normalizePageName(
			$pageName,
			'https://commons.wikimedia.org/w/api.php'
		);
		if ( $normalizedTitle === false ) {
			if ( $this->valueIncludesNamespace( $commonsLink, $namespace ) ) {
				return new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION,
					new ViolationMessage( 'wbqc-violation-message-commons-link-not-well-formed' ) );
			}
			return new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION,
				new ViolationMessage( 'wbqc-violation-message-commons-link-no-existent' ) );
		}

		return new CheckResult( $context, $constraint, CheckResult::STATUS_COMPLIANCE, null );
	}

	public function checkConstraintParameters( Constraint $constraint ): array {
		$constraintParameters = $constraint->getConstraintParameters();
		$constraintTypeItemId = $constraint->getConstraintTypeItemId();
		$exceptions = [];
		try {
			$this->constraintParameterParser->parseNamespaceParameter(
				$constraintParameters,
				$constraintTypeItemId
			);
		} catch ( ConstraintParameterException $e ) {
			$exceptions[] = $e;
		}
		return $exceptions;
	}

	private function commonsLinkIsWellFormed( string $commonsLink ): bool {
		$toReplace = [ "_", "%20" ];
		$compareString = trim( str_replace( $toReplace, '', $commonsLink ) );

		return $commonsLink === $compareString;
	}

	/**
	 * Checks whether the value of the statement already includes the namespace.
	 * This special case should be reported as “malformed title” instead of “title does not exist”.
	 */
	private function valueIncludesNamespace( string $value, string $namespace ): bool {
		return $namespace !== '' &&
			strncasecmp( $value, $namespace . ':', strlen( $namespace ) + 1 ) === 0;
	}

}
