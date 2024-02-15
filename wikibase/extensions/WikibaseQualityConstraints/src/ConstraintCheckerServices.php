<?php

namespace WikibaseQuality\ConstraintReport;

use MediaWiki\MediaWikiServices;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\LabelInLanguageChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\Lexeme\LanguageChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;

/**
 * @license GPL-2.0-or-later
 */
class ConstraintCheckerServices {

	public const CONFLICTS_WITH_CHECKER = 'WBQC_ConflictsWithChecker';
	public const ITEM_CHECKER = 'WBQC_ItemChecker';
	public const TARGET_REQUIRED_CLAIM_CHECKER = 'WBQC_TargetRequiredClaimChecker';
	public const SYMMETRIC_CHECKER = 'WBQC_SymmetricChecker';
	public const INVERSE_CHECKER = 'WBQC_InverseChecker';
	public const QUALIFIER_CHECKER = 'WBQC_QualifierChecker';
	public const QUALIFIERS_CHECKER = 'WBQC_QualifiersChecker';
	public const MANDATORY_QUALIFIERS_CHECKER = 'WBQC_MandatoryQualifiersChecker';
	public const RANGE_CHECKER = 'WBQC_RangeChecker';
	public const DIFF_WITHIN_RANGE_CHECKER = 'WBQC_DiffWithinRangeChecker';
	public const TYPE_CHECKER = 'WBQC_TypeChecker';
	public const VALUE_TYPE_CHECKER = 'WBQC_ValueTypeChecker';
	public const SINGLE_VALUE_CHECKER = 'WBQC_SingleValueChecker';
	public const MULTI_VALUE_CHECKER = 'WBQC_MultiValueChecker';
	public const UNIQUE_VALUE_CHECKER = 'WBQC_UniqueValueChecker';
	public const FORMAT_CHECKER = 'WBQC_FormatChecker';
	public const COMMONS_LINK_CHECKER = 'WBQC_CommonsLinkChecker';
	public const ONE_OF_CHECKER = 'WBQC_OneOfChecker';
	public const VALUE_ONLY_CHECKER = 'WBQC_ValueOnlyChecker';
	public const REFERENCE_CHECKER = 'WBQC_ReferenceChecker';
	public const NO_BOUNDS_CHECKER = 'WBQC_NoBoundsChecker';
	public const ALLOWED_UNITS_CHECKER = 'WBQC_AllowedUnitsChecker';
	public const SINGLE_BEST_VALUE_CHECKER = 'WBQC_SingleBestValueChecker';
	public const ENTITY_TYPE_CHECKER = 'WBQC_EntityTypeChecker';
	public const NONE_OF_CHECKER = 'WBQC_NoneOfChecker';
	public const INTEGER_CHECKER = 'WBQC_IntegerChecker';
	public const CITATION_NEEDED_CHECKER = 'WBQC_CitationNeededChecker';
	public const PROPERTY_SCOPE_CHECKER = 'WBQC_PropertyScopeChecker';
	public const CONTEMPORARY_CHECKER = 'WBQC_ContemporaryChecker';
	public const LEXEME_LANGUAGE_CHECKER = 'WBQC_Lexeme_LanguageChecker';
	public const LABEL_IN_LANGUAGE_CHECKER = 'WBQC_LabelInLanguageChecker';

	private static function getService( ?MediaWikiServices $services, $name ) {
		if ( $services === null ) {
			$services = MediaWikiServices::getInstance();
		}
		return $services->getService( $name );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getConflictsWithChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::CONFLICTS_WITH_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getItemChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::ITEM_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getTargetRequiredClaimChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::TARGET_REQUIRED_CLAIM_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getSymmetricChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::SYMMETRIC_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getInverseChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::INVERSE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getQualifierChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::QUALIFIER_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getQualifiersChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::QUALIFIERS_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getMandatoryQualifiersChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::MANDATORY_QUALIFIERS_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getRangeChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::RANGE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getDiffWithinRangeChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::DIFF_WITHIN_RANGE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getTypeChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::TYPE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getValueTypeChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::VALUE_TYPE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getSingleValueChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::SINGLE_VALUE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getMultiValueChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::MULTI_VALUE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getUniqueValueChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::UNIQUE_VALUE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getFormatChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::FORMAT_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getCommonsLinkChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::COMMONS_LINK_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getOneOfChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::ONE_OF_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getValueOnlyChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::VALUE_ONLY_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getReferenceChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::REFERENCE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getNoBoundsChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::NO_BOUNDS_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getAllowedUnitsChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::ALLOWED_UNITS_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getSingleBestValueChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::SINGLE_BEST_VALUE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getEntityTypeChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::ENTITY_TYPE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getNoneOfChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::NONE_OF_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getIntegerChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::INTEGER_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getCitationNeededChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::CITATION_NEEDED_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getPropertyScopeChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::PROPERTY_SCOPE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return ConstraintChecker
	 */
	public static function getContemporaryChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::CONTEMPORARY_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return LanguageChecker
	 */
	public static function getLexemeLanguageChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::LEXEME_LANGUAGE_CHECKER );
	}

	/**
	 * @param MediaWikiServices|null $services
	 * @return LabelInLanguageChecker
	 */
	public static function getLabelInLanguageChecker( MediaWikiServices $services = null ) {
		return self::getService( $services, self::LABEL_IN_LANGUAGE_CHECKER );
	}

}
