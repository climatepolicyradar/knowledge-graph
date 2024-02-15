<?php

/**
 * Minimal set of classes necessary to fulfill needs of parts of WikibaseQualityConstraints relying on
 * the WikibaseLexeme extension.
 */

namespace Wikibase\Lexeme\Domain\Model {

	use Wikibase\DataModel\Entity\EntityId;
	use Wikibase\DataModel\Entity\ItemId;

	class Lexeme {
		public const ENTITY_TYPE = 'lexeme';

		public function getLanguage() {
			return new ItemId( 'Q1' );
		}
	}

	class Form {
		public const ENTITY_TYPE = 'form';

	}

	class Sense {
		public const ENTITY_TYPE = 'sense';
	}

	class LexemeSubEntityId {
		public function getLexemeId() {
			return new LexemeId( 'L1' );
		}
	}

	class LexemeId extends EntityId {
		protected $serialization;

		public function __construct( string $serialization ) {
			$this->serialization = $serialization;
		}

		public function serialize() {
			return $this->serialization;
		}

		public function unserialize( $serialized ) {
			$this->serialization = $serialized;
		}

		public function getEntityType() {
			return 'lexeme';
		}
	}
}
