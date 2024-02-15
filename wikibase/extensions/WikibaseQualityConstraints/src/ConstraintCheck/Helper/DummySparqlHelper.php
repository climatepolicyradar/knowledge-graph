<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use LogicException;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Statement\Statement;

/**
 * A fake {@link SparqlHelper} that only serves as a default implementation
 * for {@link WikibaseQuality\ConstraintReport\ConstraintsServices ConstraintsServices}.
 *
 * TODO: SparqlHelper should be refactored so that this isn’t necessary.
 * See T196053#4514308 for details.
 *
 * @license GPL-2.0-or-later
 * @phan-file-suppress PhanPluginNeverReturnMethod
 */
class DummySparqlHelper extends SparqlHelper {

	public function __construct() {
		// no parent::__construct() call
	}

	public function hasType( $id, array $classes ) {
		throw new LogicException( 'methods of this class should never be called' );
	}

	public function findEntitiesWithSameStatement(
		Statement $statement,
		array $separators
	) {
		throw new LogicException( 'methods of this class should never be called' );
	}

	public function findEntitiesWithSameQualifierOrReference(
		EntityId $entityId,
		PropertyValueSnak $snak,
		$type,
		$ignoreDeprecatedStatements
	) {
		throw new LogicException( 'methods of this class should never be called' );
	}

	public function matchesRegularExpression( $text, $regex ) {
		throw new LogicException( 'methods of this class should never be called' );
	}

	public function runQuery( $query, $needsPrefixes = true ) {
		throw new LogicException( 'methods of this class should never be called' );
	}

}
