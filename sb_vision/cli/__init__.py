"""General utilities."""

import argparse
import sys

from . import debug

__all__ = ['main']


def add_subparser(subparsers, module):
    command_name = module.__name__.rsplit('.', maxsplit=1)[-1]
    parser = subparsers.add_parser(command_name, help=module.__doc__)
    module.add_arguments(parser)
    parser.set_defaults(func=module.main)


def argument_parser():
    parser = argparse.ArgumentParser(
        __name__.rsplit('.', maxsplit=1)[0],
        description="General utilities for the sb-vision library",
    )

    subparsers = parser.add_subparsers()

    add_subparser(subparsers, debug)

    return parser


def main(args=None):
    """Main entry point."""
    if args is None:
        args = sys.argv[1:]

    parser = argument_parser()
    options = parser.parse_args(args)

    if 'func' in options:
        settings = vars(options)
        settings.pop('func')(**settings)
    else:
        parser.print_help()

