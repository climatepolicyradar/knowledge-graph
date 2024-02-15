<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Context;

/**
 * A serializer for {@link ContextCursor}s.
 *
 * @license GPL-2.0-or-later
 */
class ContextCursorSerializer {

	/**
	 * @param ContextCursor $cursor
	 * @return array
	 */
	public function serialize( ContextCursor $cursor ) {
		if ( $cursor instanceof EntityContextCursor ) {
			return [
				't' => '\entity',
				'i' => $cursor->getEntityId(),
			];
		}

		$type = $cursor->getType();
		$serialization = [
			't' => $type,
			'i' => $cursor->getEntityId(),
			'p' => $cursor->getStatementPropertyId(),
			'g' => $cursor->getStatementGuid(),
			'h' => $cursor->getSnakHash(),
		];

		if ( $type === Context::TYPE_QUALIFIER || $type === Context::TYPE_REFERENCE ) {
			$serialization['P'] = $cursor->getSnakPropertyId();
			if ( $type === Context::TYPE_REFERENCE ) {
				/** @var ReferenceContextCursor $cursor */
				'@phan-var ReferenceContextCursor $cursor';
				$serialization['r'] = $cursor->getReferenceHash();
			}
		}

		return $serialization;
	}

}
