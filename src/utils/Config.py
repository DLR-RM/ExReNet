import json


class Config:

    def __init__(self, config):
        """Creates a configuration block.

        Args:
            data(dict): A nested key/value structure of raw configurations.
        """
        self.prefix = ""
        self.config = config
        self.printed_settings = {}

    @staticmethod
    def from_file(path):
        with open(path, "r") as f:
            return Config(json.load(f))

    def get_with_prefix(self, prefix):
        """Clones this config and adds an additional prefix.

        Args:
            prefix(str): The additional prefix.

        Returns:
            Configuration: The new config with the requested prefix.
        """
        config = Config(self.config)
        config.prefix = self.prefix + prefix + "/"
        return config


    def _get_value_from_block(self, name, block):
        if "/" in name:
            delimiter_pos = name.find("/")
            block_name = name[:delimiter_pos]
            if block_name in block and type(block[block_name]) is dict:
                return self._get_value_from_block(name[delimiter_pos + 1:], block[block_name])
            else:
                raise NotFoundError("No such configuration block '" + block_name + "'!")
        else:
            if name in block:
                return block[name]
            else:
                raise NotFoundError("No such configuration '" + name + "'!")

    def _get_value(self, name):
        """Returns the configuration with the given name and the given type.

        If the configuration cannot be found in this config, the request will use the base config as fallback.

        Args:
            name(str): The name of the configuration.
            value_type(str): The desired type of the configuration value.

        Returns:
            object: The value of the configuration in the requested type.

        Raises:
            NotFoundError: If the configuration cannot be found in this or any base configs.
            TypeError. If the configuration value cannot be converted into the requested type.
        """
        return self._get_value_from_block(self.prefix + name, self.config)

    def _get_value_with_fallback(self, name, fallback):
        """Returns the configuration with the given name or the fallback name and the given type.

        If the configuration cannot be found in this config, the request will use the base config as fallback.
        If there is no configuration with the given name, the fallback configuration will be used.

        Args:
            name(str): The name of the configuration.
            fallback(str): The name of the fallback configuration, which should be used if the primary one cannot be found.
            value_type(str): The desired type of the configuration value.

        Returns:
            object: The value of the configuration in the requested type.

        Raises:
            NotFoundError: If the configuration cannot be found in this or any base configs.
            TypeError. If the configuration value cannot be converted into the requested type.
        """

        try:
            value = self._get_value(name)
        except NotFoundError:
            if fallback is not None:
                value = self._get_value(fallback)
            else:
                raise

        if name not in self.printed_settings:
            if name not in self.printed_settings:
                print("Using " + name + " = " + str(value))
            self.printed_settings[name] = value

        return value

    def get_int(self, name, fallback=None):
        """Returns the value of the configuration with the given name as integer.

        Args:
            name(str): The name of the configuration.

        Returns:
            int: The integer value of the configuration.

        Raises:
            TypeError: If the value could not be converted to an integer.
        """
        value = self._get_value_with_fallback(name, fallback)
        try:
            return int(value)
        except ValueError:
            raise TypeError("Cannot convert '" + str(value) + "' to int!")

    def get_bool(self, name, fallback=None):
        """Returns the value of the configuration with the given name as bool.

        Args:
            name(str): The name of the configuration.

        Returns:
            bool: The bool value of the configuration.

        Raises:
            TypeError: If the value could not be converted to a bool.
        """
        value = self._get_value_with_fallback(name, fallback)
        try:
            return bool(value)
        except ValueError:
            raise TypeError("Cannot convert '" + str(value) + "' to bool!")

    def get_float(self, name, fallback=None):
        """Returns the value of the configuration with the given name float.

        Args:
            name(str): The name of the configuration.

        Returns:
            float: The float value of the configuration.

        Raises:
            TypeError: If the value could not be converted to a float.
        """
        value = self._get_value_with_fallback(name, fallback)
        try:
            return float(value)
        except ValueError:
            raise TypeError("Cannot convert '" + str(value) + "' to float!")

    def get_string(self, name, fallback=None):
        """Returns the value of the configuration with the given name as string.

        Args:
            name(str): The name of the configuration.

        Returns:
            string: The string value of the configuration.

        Raises:
            TypeError: If the value could not be converted to a string.
        """
        value = self._get_value_with_fallback(name, fallback)
        try:
            return str(value)
        except ValueError:
            raise TypeError("Cannot convert '" + str(value) + "' to string!")

    def get_list(self, name, fallback=None):
        """Returns the value of the configuration with the given name as list.

        Args:
            name(str): The name of the configuration.

        Returns:
            list: The list value of the configuration.

        Raises:
            TypeError: If the value could not be converted to a string.
        """
        value = self._get_value_with_fallback(name, fallback)
        if not isinstance(value, list):
            raise TypeError("Cannot convert '" + str(value) + "' to list!")
        return value

class NotFoundError(Exception):
    pass
