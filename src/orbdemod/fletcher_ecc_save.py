from typing import List
from .logging_config import get_module_logger

logger = get_module_logger(__name__)


# ===========================
# 1. Basic Functions
# ===========================


def fletcher_checksum(
    hex_data: str
) -> str:
    """
    Calculate the fletcher checksum of the input hex string.

    Args:
        hex_data: parsed ORBCOMM packets hex string, including FCS words.
    
    Returns:
        str: two checksum words as uppercase hex (sum2, sum1).
    """
    sum1 = 0
    sum2 = 0

    for xx in range(0, len(hex_data), 2):
        val = int(hex_data[xx:xx+2], 16)
        sum1 = (sum1 + val) % 256
        sum2 = (sum1 + sum2) % 256

    return f"{sum2:02X}{sum1:02X}"


def single_bit_fix(
    hex_data: str
) -> str | None:
    """
    Attempt single bit error correction on a single packet.

    Args:
        hex_data: parsed ORBCOMM packets hex string, including FCS words.
    
    Returns:
        str | None: if the error correction is successful, return the corrected hex string; otherwise, return none.
    """
    num_hex_data = len(hex_data)
    num_bits = num_hex_data * 4

    binary_packet = f"{int(hex_data, 16):0{num_bits}b}"

    for i in range(len(binary_packet)):
        original_bit = binary_packet[i]
        flip_bit = '1' if original_bit == '0' else '0'

        temp_bits = binary_packet[:i] + flip_bit + binary_packet[i+1:]

        temp_hex = f"{int(temp_bits, 2):0{num_hex_data}X}"

        if fletcher_checksum(temp_hex) == '0000':
            return temp_hex
    return None


# ===========================
# 2. Composite Function
# ===========================




def validate_packet(
    hex_data: List[str],
    output_file: str | None = "orbdemod_packets.txt" ,
) -> List[str]:
    """
    Verify the fletcher checksum of the hex packet, attempt single-bit error correction, and save the result.

    Args:
        hex_data: parsed ORBCOMM packets hex string, including FCS words.
        output_file: the storage path of the valid packets; if it is None, the file will not be saved.

    Returns:
        valid_packets: List of packets that have passed the fletcher checksum or have been corrected.

    """
    logger.info("Packets integrity check (fletcher checksum), 1-bit ECC and save the file.")

    valid_packets = []
    fixed_count = 0
    error_count = 0
    total_packets = len(hex_data)

    for packet in hex_data:
        status = "FAIL"
        final_packet = packet

        if fletcher_checksum(packet) == '0000':
            status = "OK"
        else:
            corrected_packet = single_bit_fix(packet)

            if corrected_packet is not None:
                final_packet = corrected_packet
                status = "FIXED"
                fixed_count +=1
        if status == "FAIL":
            error_count +=1
            logger.warning("Fletcher checksum fails and ECC is unsuccessful or disabled. Dropping packet: %s.", packet)
        else:
            valid_packets.append(final_packet)
            if status == "FIXED":
                logger.info("Packet successfully fix: %s.", final_packet)
    if output_file is not None:
        logger.info("Writing valid and corrected packets to: %s.", output_file)

        with open(output_file, 'w')as f_out:
            for packet in valid_packets:
                f_out.write(packet + '\n')

    valid_count = len(valid_packets)
    ok_count = valid_count - fixed_count

    per = 0.0
    if total_packets > 0:
        per = float(error_count) / total_packets * 100

    
    
    logger.info("_" * 50)
    logger.info("Fletcher checksum and 1-bit ECC processing has completed.")
    logger.info("Total packets processed: %d.", total_packets)
    logger.info("Successfully validated/fixed packets: %d.", ok_count + fixed_count)
    logger.info("Packets dropped: %.2f%% (%d).", per, error_count)
    if output_file is None:
        logger.info("Packet saving is skipped as output_file is set to None.")
    logger.info("-" * 50)

    return valid_packets