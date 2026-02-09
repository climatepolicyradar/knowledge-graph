"""Fetch pricing information from OpenRouter API."""

import logging
from typing import Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelPricing(BaseModel):
    """
    Pricing information for a model.

    https://openrouter.ai/docs/guides/overview/models#pricing-object
    """

    prompt_price: float = Field(
        description="The price per prompt token in USD.",
    )
    completion_price: float = Field(
        description="The price per completion token in USD."
    )


async def get_openrouter_pricing(model_name: str) -> Optional[ModelPricing]:
    """
    Fetch pricing information for an OpenRouter model.

    :param model_name: The model name (e.g., "openrouter:openai/gpt-5" or "openai/gpt-5")
    :type model_name: str
    :return: Pricing information if successful, None on any error
    :rtype: Optional[ModelPricing]
    """

    if model_name.startswith("openrouter:"):
        model_name = model_name[len("openrouter:") :]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()

            data = response.json()

            models = data.get("data", [])
            for model in models:
                if model.get("id") == model_name:
                    pricing = model.get("pricing", {})
                    prompt = pricing.get("prompt")
                    completion = pricing.get("completion")

                    if prompt is not None and completion is not None:
                        prompt_price = float(prompt)
                        completion_price = float(completion)

                        logger.info(
                            f"Fetched pricing for {model_name}: prompt=${prompt_price}, completion=${completion_price}"
                        )

                        return ModelPricing(
                            prompt_price=prompt_price,
                            completion_price=completion_price,
                        )

            logger.warning(f"Model {model_name} not found in OpenRouter API response")
            return None

    except httpx.HTTPError as e:
        logger.warning(f"HTTP error fetching OpenRouter pricing: {e}")
        return None
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Error parsing OpenRouter pricing response: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching OpenRouter pricing: {e}")
        return None
