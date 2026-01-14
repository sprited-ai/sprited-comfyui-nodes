# file: parse_int.py
#
# SpriteDX_ParseInt – Convert string to integer
# -----------------------------------------------
#   Takes a string and converts it to an integer.
#
# Inputs:
#   • string_value: String to parse
#   • default_value: Default value if parsing fails
#
# Outputs:
#   • int_value: Parsed integer

class SpriteDXParseIntNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_value": ("STRING", {"default": "0"}),
            },
            "optional": {
                "default_value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int_value",)
    FUNCTION = "parse_int"
    CATEGORY = "utils"
    DESCRIPTION = "Convert string to integer."

    def parse_int(self, string_value, default_value=0):
        """
        Parse string to integer.
        
        Args:
            string_value: String to parse
            default_value: Value to return if parsing fails
            
        Returns:
            Parsed integer or default value
        """
        try:
            return (int(string_value),)
        except (ValueError, TypeError):
            return (default_value,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SpriteDX_ParseInt": SpriteDXParseIntNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpriteDX_ParseInt": "Parse Int (SpriteDX)",
}
