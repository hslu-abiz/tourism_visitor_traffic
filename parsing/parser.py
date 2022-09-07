# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 21.11.19
#
import argparse
from typing import Any, Dict, List, Optional


def _convert_arg_line_to_args(arg_line: str) -> List[str]:
    args = arg_line.split()
    # Ignore empty lines and comments
    if not args or args[0].startswith('#'):
        return list()
    # Add '--' in front of the first option per line if not present
    if not args[0].startswith('-'):
        args[0] = '--' + args[0]
    return args


class Parser(object):

    def __init__(
            self,
            description: str = '',
            argument_default: str = argparse.SUPPRESS,
            fromfile_prefix_chars: str = '@',
            add_help: bool = False,
            **kwargs,
    ):
        self.parser = argparse.ArgumentParser(
            description=description,
            argument_default=argument_default,
            fromfile_prefix_chars=fromfile_prefix_chars,
            add_help=add_help,
            **kwargs,
        )
        self.parser.convert_arg_line_to_args = _convert_arg_line_to_args
        self.arguments_by_group = {}
        self.groups = {}
        self.args = None

    def add_group(
            self,
            group_name: str,
            arguments: Optional[Dict[str, Dict[str, Any]]] = None,
            group_suffix: str = ' arguments',
    ) -> None:
        if arguments is None:
            arguments = {}
        if group_name in self.groups.keys():
            raise ValueError('Group {} already exists.'.format(group_name))
        group = self.parser.add_argument_group(group_name + group_suffix)
        self.groups[group_name] = group
        for argument_name in arguments.keys():
            group.add_argument('--' + argument_name, **arguments[argument_name])
        self.arguments_by_group[group_name] = list(arguments.keys())

    def add_argument(
            self,
            group_name: str,
            argument_name: str,
            *args,
            **kwargs,
    ) -> None:
        if group_name not in self.groups.keys():
            self.add_group(group_name)
        self.groups[group_name].add_argument('--' + argument_name, *args, **kwargs)
        self.arguments_by_group[group_name].append(argument_name)

    def parse(self, *args, **kwargs) -> None:
        self.args = self.parser.parse_args(*args, **kwargs)

    def parse_known(self, *args, **kwargs) -> list:
        """Parses known and return unknown arguments."""
        self.args, unknown = self.parser.parse_known_args(*args, **kwargs)
        return unknown

    def get_group(self, group_name: str) -> Dict[str, Any]:
        return {k: v for k, v in vars(self.args).items()
                if k in self.arguments_by_group[group_name]}

    def get_argument(self, argument_name: str) -> Any:
        try:
            return self.args[argument_name]
        except KeyError:
            raise KeyError('Argument {} not found.'.format(argument_name))
