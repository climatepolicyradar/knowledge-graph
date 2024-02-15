<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Result;

use DomainException;
use Wikibase\DataModel\Entity\NumericPropertyId;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursor;

/**
 * A blank CheckResult that only holds a context cursor, but no actual result.
 * Used for contexts that should appear in the API output
 * even if no constraints are defined for them.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 * @phan-file-suppress PhanPluginNeverReturnMethod
 */
class NullResult extends CheckResult {

	/**
	 * The property ID that is used for the fake constraint passed to the superclass.
	 * Since the NumericPropertyId constructor prevents invalid property IDs like “P0”,
	 * we use the maximum permitted property ID and assume that it’s unlikely to actually exist.
	 */
	private const NULL_PROPERTY_ID = 'P2147483647';

	public function __construct( ContextCursor $contextCursor ) {
		$constraint = new Constraint(
			'null',
			new NumericPropertyId( self::NULL_PROPERTY_ID ),
			'none',
			[]
		);
		parent::__construct( $contextCursor, $constraint );
	}

	public function getConstraint(): Constraint {
		throw new DomainException( 'NullResult holds no constraint' );
	}

	public function getConstraintId(): string {
		throw new DomainException( 'NullResult holds no constraint' );
	}

}
