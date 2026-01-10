import numpy as np
from typing import Tuple, List
from .logging_config import get_module_logger

logger = get_module_logger(__name__)

# from https://github.com/fbieberly/ORBCOMM-receiver/blob/master/orbcomm_packet.py

class OrbcommPacketType:
    DEFAULT_PACKET_SIZE_BITS = 12 * 8

    packet_dict = {
        'Sync':{
            'header':'01100101',
            'hex_header':'65',
            'message_parts':[
                ('code',(0,6)),
                ('sat_id',(6,8)),
            ],
        },

        'Message':{
            'header':'00011010',
            'hex_header':'1A',
            'message_parts':[
                ('msg_total_length', (2, 3)), 
                ('msg_packet_num', (3, 4)),
                ('data', (4, 20)),
            ],
        },

        'Uplink_info':{
            'header':'00011011',
            'hex_header':'1B',
            'message_parts':[
                ('msg_total_length', (2, 3)),
                ('msg_packet_num', (3, 4)),
                ('data', (4, 20)),
            ],
        },

        'Downlink_info':{
            'header':'00011100',
            'hex_header':'1C',
            'message_parts':[
                ('msg_total_length', (2, 3)),
                ('msg_packet_num', (3, 4)),
                ('data', (4, 20)),
            ],
        },

        'Network':{
            'header':'00011101',
            'hex_header':'1D',
            'message_parts':[
                ('msg_total_length', (2, 3)),
                ('msg_packet_num', (3, 4)),
                ('data', (4, 20)),
            ],

        },

        'Fill':{
            'header':'00011110',
            'hex_header':'1E',
            'message_parts':[ 
                ('data', (2, 20)),
            ],

        },

        'Ephemeris':{
            'header':'00011111',
            'hex_header':'1F',
            'message_parts':[
                ('sat_id', (2, 4)),  
                ('data', (4, 46)),  
            ],

        },

        'Orbital':{
            'header':'00100010',
            'hex_header':'22',
            'message_parts':[
            ],

        },
    }


    @staticmethod
    def get_packet_headers() -> List[str]:
        """
        Extract binary header strings for all known data packets.
        """
        return [data['header'] for data in OrbcommPacketType.packet_dict.values()]


def find_packet_start(
    decode_bits: np.ndarray
) -> Tuple[int, bool]:
    """
    The best packet starting point and byte order are determined by calculating the number of matches with the known headers on all possible offset.

    Args:
        decode_bits: the demodulated bit data (not string yet).

    Return:
        best_offset: the optimal starting bit offset.
        use_reversed_bits: True means bits are reversed (LSB-first).
    """
    bit_string = ''.join(map(str, decode_bits))
    logger.info("Starting search for packets best offset and bits order.")

    packet_headers = OrbcommPacketType.get_packet_headers()
    packet_size_bits = OrbcommPacketType.DEFAULT_PACKET_SIZE_BITS


    normal_scores = np.zeros(packet_size_bits)
    rev_scores = np.zeros(packet_size_bits)

    for offset in range(packet_size_bits):
        for i in range(offset, len(bit_string) - 8, packet_size_bits):
            header_candidate = bit_string[i : i+8]
            
            if header_candidate[::-1] in packet_headers:
                rev_scores[offset] += 1
            if header_candidate in packet_headers:
                normal_scores[offset] += 1
    
    use_reversed_bits = np.max(rev_scores) > np.max(normal_scores)

    if use_reversed_bits:
        best_offset = np.argmax(rev_scores)
        logger.info(
            "Optimal offset found at: %d (Bits are REVERSED).", 
            best_offset
        )
    else:
        best_offset = np.argmax(normal_scores)
        logger.info(
            "Optimal offset found at: %d (Bits are NOT reversed).", 
            best_offset
        )

    return best_offset, use_reversed_bits



def bits_to_packets(
    decode_bits: np.ndarray,
    offset: int,
    reverse_order: bool
) -> List[str]:
    """
    Parse Orbcomm packets from the bit stream based on the best offset and bits order.

    Args:
        decode_bits: the demodulated bit data (not string yet).
        offset: the optimal starting bit offset.
        reverse_order: True means bits are reversed (LSB-first).

    Returns:
        List[str]: the list of parsed packets, each items is a hex string.
    """
    bit_string = ''.join(map(str, decode_bits))
    logger.info("Parsing bits stream into hex packets.")

    packets = []

    protocol_dict = OrbcommPacketType.packet_dict
    packet_size_bits = OrbcommPacketType.DEFAULT_PACKET_SIZE_BITS


    eph_header = protocol_dict['Ephemeris']['hex_header']
    i = offset

    while i <= len(bit_string) - packet_size_bits:
        header_bits = bit_string[i : i+8]
        if reverse_order:
            header_bits = header_bits[::-1]


        header_hex = f"{int(header_bits, 2):02X}"
        current_packet_len_bits = packet_size_bits
        
        if header_hex == eph_header:
            current_packet_len_bits = 24 * 8
            logger.debug("Detect ephemeris packet (Type 1F).")
        
        if i + current_packet_len_bits > len(bit_string):
            break
            
        packet_hex = ""
        for j in range(0, current_packet_len_bits, 8):
            byte_bits = bit_string[i+j : i+j+8]
            if reverse_order:
                byte_bits = byte_bits[::-1]
            packet_hex += f"{int(byte_bits, 2):02X}"
        
        packets.append(packet_hex)
        
        i += current_packet_len_bits
        
    logger.info("Successfully parse %d packets from the bits stream.", len(packets))
    
    return packets