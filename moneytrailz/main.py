import logging

import click
import click_log

logger = logging.getLogger(__name__)
click_log.basic_config(logger)  # type: ignore


CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"], auto_envvar_prefix="MONEYTRAILZ"
)


@click.command(context_settings=CONTEXT_SETTINGS)
@click_log.simple_verbosity_option(logger)  # type: ignore
@click.option(
    "-c",
    "--config",
    help="Path to toml config",
    required=True,
    default="moneytrailz.toml",
    type=click.Path(exists=True, readable=True),
)
@click.option(
    "--without-ibc",
    is_flag=True,
    help="Run without IBC. Enable this if you want to run the TWS "
    "gateway yourself, without having MoneyTrailz manage it for you.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform a dry run. This will display the the orders without sending any live trades.",
)
def cli(config: str, without_ibc: bool, dry_run: bool) -> None:
    """MoneyTrailz is an IBKR bot for collecting money.

    You can configure this tool by supplying a toml configuration file.
    Evolved from thetagang with advanced strategy framework and real-time dashboard.
    """

    from .moneytrailz import start

    start(config, without_ibc, dry_run)
