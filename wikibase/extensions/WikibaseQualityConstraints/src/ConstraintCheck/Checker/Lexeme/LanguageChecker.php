<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\Lexeme;

use Wikibase\DataModel\Entity\EntityDocument;
use Wikibase\DataModel\Services\Lookup\EntityLookup;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\Lexeme\Domain\Model\Form;
use Wikibase\Lexeme\Domain\Model\Lexeme;
use Wikibase\Lexeme\Domain\Model\LexemeSubEntityId;
use Wikibase\Lexeme\Domain\Model\Sense;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @license GPL-2.0-or-later
 */
class LanguageChecker implements ConstraintChecker {

	/**
	 * @var EntityLookup
	 */
	private $entityLookup;

	/**
	 * @var ConstraintParameterParser
	 */
	private $constraintParameterParser;

	public function __construct(
		ConstraintParameterParser $constraintParameterParser,
		EntityLookup $lookup
	) {
		$this->constraintParameterParser = $constraintParameterParser;
		$this->entityLookup = $lookup;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getSupportedContextTypes() {
		return self::ALL_CONTEXT_TYPES_SUPPORTED;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getDefaultContextTypes() {
		return Context::ALL_CONTEXT_TYPES;
	}

	/** @codeCoverageIgnore This method is purely declarative. */
	public function getSupportedEntityTypes() {
		return [
			'item' => CheckResult::STATUS_NOT_IN_SCOPE,
			'property' => CheckResult::STATUS_NOT_IN_SCOPE,
			'lexeme' => CheckResult::STATUS_COMPLIANCE,
			'form' => CheckResult::STATUS_COMPLIANCE,
			'sense' => CheckResult::STATUS_COMPLIANCE,
			'mediainfo' => CheckResult::STATUS_NOT_IN_SCOPE,
		];
	}

	/**
	 * Checks 'Language' constraint.
	 *
	 * @param Context $context
	 * @param Constraint $constraint
	 *
	 * @throws ConstraintParameterException
	 * @return CheckResult
	 */
	public function checkConstraint( Context $context, Constraint $constraint ) {
		if ( $context->getSnakRank() === Statement::RANK_DEPRECATED ) {
			return new CheckResult( $context, $constraint, CheckResult::STATUS_DEPRECATED );
		}

		$constraintParameters = $constraint->getConstraintParameters();
		$constraintTypeItemId = $constraint->getConstraintTypeItemId();

		$languages = $this->constraintParameterParser->parseItemsParameter(
			$constraintParameters,
			$constraintTypeItemId,
			true
		);

		$message = ( new ViolationMessage( 'wbqc-violation-message-language' ) )
			->withEntityId( $context->getSnak()->getPropertyId(), Role::PREDICATE )
			->withItemIdSnakValueList( $languages, Role::OBJECT );
		$status = CheckResult::STATUS_VIOLATION;

		$lexeme = $this->getLexeme( $context );
		if ( !$lexeme ) {
			// Lexeme doesn't exist, let's not bother
			return new CheckResult( $context, $constraint, CheckResult::STATUS_NOT_IN_SCOPE );
		}

		/** @var Lexeme $lexeme */
		'@phan-var Lexeme $lexeme';

		foreach ( $languages as $language ) {
			if ( $language->isNoValue() || $language->isSomeValue() ) {
				continue;
			}
			if ( $lexeme->getLanguage()->equals( $language->getItemId() ) ) {
				$message = null;
				$status = CheckResult::STATUS_COMPLIANCE;
				break;
			}
		}

		return new CheckResult( $context, $constraint, $status, $message );
	}

	private function getLexeme( Context $context ): ?EntityDocument {
		$entityType = $context->getEntity()->getType();

		if ( $entityType === Lexeme::ENTITY_TYPE ) {
			return $context->getEntity();
		}

		if ( in_array( $entityType, [ Form::ENTITY_TYPE, Sense::ENTITY_TYPE ] ) ) {
			/** @var LexemeSubEntityId $id */
			$id = $context->getEntity()->getId();
			'@phan-var LexemeSubEntityId $id';
			return $this->entityLookup->getEntity( $id->getLexemeId() );
		}
	}

	public function checkConstraintParameters( Constraint $constraint ): array {
		$constraintParameters = $constraint->getConstraintParameters();
		$constraintTypeItemId = $constraint->getConstraintTypeItemId();
		$exceptions = [];
		try {
			$this->constraintParameterParser->parseItemsParameter(
				$constraintParameters,
				$constraintTypeItemId,
				true
			);
		} catch ( ConstraintParameterException $e ) {
			$exceptions[] = $e;
		}
		return $exceptions;
	}

}
