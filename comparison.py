import parselmouth
from praatio import textgrid
import pickle
import os
import numpy as np
from models import consonant_classifiers, vowel_classifiers, voicing_classifiers


consonants = set("b bʲ c d dz dʑ dʲ h j k m mʲ n p pʲ s t ts tɕ tʲ v vʲ w z ç ŋ ɕ ɟ ɡ ɰ̃ ɲ ɴ ɸ ɸʲ ɾ ɾʲ ʑ ʔ bː cː dzː dʑː dː hː kː mʲː mː nː pʲː pː sː tsː tɕː tː çː ɕː ɡː ɲː ɴː ɸː ɾː".split())
vowels = set("a e i o ɨ ɯ aː eː iː oː ɨː ɯː i̥ ɨ̥ ɯ̥".split())
voiceless_stops = set(['p', 'pʲ', 't', 'tʲ', 'k', 'c', 'pː', 'pʲː', 'tː', 'tʲː', 'kː', 'cː'])
voiced_stops = set(['b', 'bʲ', 'd', 'dʲ', 'g', 'ɡ', 'ɟ', 'bː', 'bʲː', 'dː', 'dʲː', 'gː', 'ɡː', 'ɟː'])

voicing_pairs = {
    'p': {'voiceless': 'p', 'voiced': 'b'},
    'pʲ': {'voiceless': 'pʲ', 'voiced': 'bʲ'},
    't': {'voiceless': 't', 'voiced': 'd'},
    'tʲ': {'voiceless': 'tʲ', 'voiced': 'dʲ'},
    'k': {'voiceless': 'k', 'voiced': 'g'},
    'c': {'voiceless': 'c', 'voiced': 'ɟ'},
    'pː': {'voiceless': 'pː', 'voiced': 'bː'},
    'pʲː': {'voiceless': 'pʲː', 'voiced': 'bʲː'},
    'tː': {'voiceless': 'tː', 'voiced': 'dː'},
    'tʲː': {'voiceless': 'tʲː', 'voiced': 'dʲː'},
    'kː': {'voiceless': 'kː', 'voiced': 'gː'},
    'cː': {'voiceless': 'cː', 'voiced': 'ɟː'},
    'b': {'voiceless': 'p', 'voiced': 'b'},
    'bʲ': {'voiceless': 'pʲ', 'voiced': 'bʲ'},
    'd': {'voiceless': 't', 'voiced': 'd'},
    'dʲ': {'voiceless': 'tʲ', 'voiced': 'dʲ'},
    'g': {'voiceless': 'k', 'voiced': 'g'},
    'ɡ': {'voiceless': 'k', 'voiced': 'ɡ'},
    'ɟ': {'voiceless': 'c', 'voiced': 'ɟ'},
    'bː': {'voiceless': 'pː', 'voiced': 'bː'},
    'bʲː': {'voiceless': 'pʲː', 'voiced': 'bʲː'},
    'dː': {'voiceless': 'tː', 'voiced': 'dː'},
    'dʲː': {'voiceless': 'tʲː', 'voiced': 'dʲː'},
    'gː': {'voiceless': 'kː', 'voiced': 'gː'},
    'ɡː': {'voiceless': 'kː', 'voiced': 'ɡː'},
    'ɟː': {'voiceless': 'cː', 'voiced': 'ɟː'},
}

def get_base_phoneme(phoneme):
    base = phoneme.replace('ː', '')
    base = base.replace('ʲ', '')
    return base

def get_phoneme_type(phoneme):
    if phoneme in consonants:
        return 'consonant'
    elif phoneme in vowels:
        return 'vowel'
    return None

def check_voicing_error(intended_phoneme, vot_ms):

    if intended_phoneme not in voicing_pairs:
        return None

    pair_info = voicing_pairs[intended_phoneme]
    pair_key_full = pair_info['voiceless']
    pair_key = get_base_phoneme(pair_key_full) 

    if pair_key not in voicing_classifiers:
        is_intended_voiceless = intended_phoneme in voiceless_stops
        perceived_as_voiceless = vot_ms > 25 
        if is_intended_voiceless != perceived_as_voiceless:
            perceived = pair_info['voiceless'] if perceived_as_voiceless else pair_info['voiced']
            return {
                'intended': intended_phoneme,
                'vot': vot_ms,
                'boundary': 25,
                'perceived_as': perceived,
                'is_error': True,
                'severity': 'Warning',
                'prob_voiceless': 1.0 if vot_ms > 25 else 0.0,
                'prob_voiced': 0.0 if vot_ms > 25 else 1.0,
                'z_score': 0
            }
        return None  # No error using fallback

    info = voicing_classifiers[pair_key]
    clf = info['classifier']

    is_intended_voiceless = intended_phoneme in voiceless_stops

    prob_voiced = clf.predict_proba([[vot_ms]])[0][1]
    prob_voiceless = 1 - prob_voiced

    perceived_as_voiceless = vot_ms > info['boundary']
    is_error = (is_intended_voiceless != perceived_as_voiceless)

    expected_mean = info['voiceless_mean'] if is_intended_voiceless else info['voiced_mean']
    expected_std = info['voiceless_std'] if is_intended_voiceless else info['voiced_std']
    z_score = (vot_ms - expected_mean) / expected_std if expected_std > 0 else 0

    if abs(z_score) < 1:
        severity = 'OK'
    elif abs(z_score) < 2:
        severity = 'Warning'
    else:
        severity = 'Error'

    perceived_phoneme = info['voiceless_phoneme'] if perceived_as_voiceless else info['voiced_phoneme']

    return {
        'intended': intended_phoneme,
        'vot': vot_ms,
        'boundary': info['boundary'],
        'prob_voiceless': prob_voiceless,
        'prob_voiced': prob_voiced,
        'perceived_as': perceived_phoneme,
        'is_error': is_error,
        'z_score': z_score,
        'severity': severity
    }

def check_length_error(intended_phoneme, actual_duration_ms):

    phoneme_type = get_phoneme_type(intended_phoneme)
    if phoneme_type is None:
        return None

    base = get_base_phoneme(intended_phoneme)
    is_intended_long = 'ː' in intended_phoneme

    classifiers = consonant_classifiers if phoneme_type == 'consonant' else vowel_classifiers

    if base not in classifiers:
        return None

    info = classifiers[base]
    clf = info['classifier']

    prob_long = clf.predict_proba([[actual_duration_ms]])[0][1]
    prob_short = 1 - prob_long

    perceived_as_long = actual_duration_ms > info['boundary']
    is_error = (is_intended_long != perceived_as_long)

    expected_mean = info['long_mean'] if is_intended_long else info['short_mean']
    expected_std = info['long_std'] if is_intended_long else info['short_std']
    z_score = (actual_duration_ms - expected_mean) / expected_std if expected_std > 0 else 0

    if abs(z_score) < 1:
        severity = 'OK'
    elif abs(z_score) < 2:
        severity = 'Warning'
    else:
        severity = 'Error'

    return {
        'intended': intended_phoneme,
        'duration': actual_duration_ms,
        'boundary': info['boundary'],
        'prob_perceived_as_short': prob_short,
        'prob_perceived_as_long': prob_long,
        'perceived_as': f'{base}ː' if perceived_as_long else base,
        'is_error': is_error,
        'z_score': z_score,
        'severity': severity
    }

def get_info(audio_path, textgrid_path, tier_name="phones"):
    snd = parselmouth.Sound(audio_path)

    def get_phoneme_segments(tg_path, tier_name):
        tg = textgrid.openTextgrid(tg_path, True)
        entries = tg._tierDict[tier_name].entries
        return [(start, end, label) for start, end, label in entries if label.strip()]

    def calculate_vot(start, end):
        segment_duration = (end - start) * 1000
        if segment_duration < 30:
            return None

        segment = snd.extract_part(from_time=start, to_time=end, preserve_times=True)

        try:
            time_step = min(0.005, (end - start) / 10)
            intensity = segment.to_intensity(time_step=time_step)
            intensity_values = intensity.values[0]
            intensity_times = intensity.xs()

            if len(intensity_values) == 0:
                return None

            burst_idx = np.argmax(intensity_values)
            burst_time = intensity_times[burst_idx]

            pitch = segment.to_pitch(
                time_step=time_step,
                pitch_floor=75.0,
                pitch_ceiling=600.0
            )

            voicing_onset_time = None
            for t in pitch.xs():
                from parselmouth.praat import call
                pitch_value = call(pitch, "Get value at time", t, "Hertz", "Linear")
                if pitch_value > 0:
                    voicing_onset_time = t
                    break

            if voicing_onset_time is not None:
                vot = (voicing_onset_time - burst_time) * 1000
                return vot
            return None
        except:
            return None

    segments = get_phoneme_segments(textgrid_path, tier_name)
    results = []
    for start, end, label in segments:
        duration = end - start
        vot = None
        if label in voiceless_stops or label in voiced_stops:
            vot = calculate_vot(start, end)
        results.append((label, duration, vot))
    return results


def compare_infos(user_wav, user_textgrid, ref_wav, ref_textgrid):
    user_info = get_info(user_wav, user_textgrid)
    ref_info = get_info(ref_wav, ref_textgrid)
    comparisons = []

    for user_data, ref_data in zip(user_info, ref_info):
        u_label, u_dur, u_vot = user_data
        r_label, r_dur, r_vot = ref_data

        if u_label == r_label:
            diff_length = u_dur - r_dur
            u_dur_ms = u_dur * 1000

            comparison = f"{u_label}: "
            is_error = False

            error_result = check_length_error(u_label, u_dur_ms)

            if error_result:
                if error_result['is_error']:
                    is_error = True
                    comparison += f"LENGTH: Perceived as /{error_result['perceived_as']}/ "
                    comparison += f"({error_result['severity']}, "
                    comparison += f"{error_result['prob_perceived_as_long']:.0%} prob long)"
                elif error_result['severity'] == 'Warning':
                    is_error = True
                    comparison += f"LENGTH: {error_result['severity']} "
                    comparison += f"(z={error_result['z_score']:.1f})"

            is_stop = u_label in voiceless_stops or u_label in voiced_stops

            if is_stop:
                if u_vot is not None:
                    voicing_result = check_voicing_error(u_label, u_vot)

                    if voicing_result is not None:
                        if voicing_result['is_error']:
                            is_error = True
                            if comparison != f"{u_label}: ":
                                comparison += " | "
                            comparison += f"VOICING: Perceived as /{voicing_result['perceived_as']}/ "
                            comparison += f"({voicing_result['severity']}, "
                            comparison += f"{voicing_result['prob_voiceless']:.0%} prob voiceless)"
                        elif voicing_result['severity'] == 'Warning':
                            is_error = True
                            if comparison != f"{u_label}: ":
                                comparison += " | "
                            comparison += f"VOICING: {voicing_result['severity']} "
                            comparison += f"(z={voicing_result['z_score']:.1f})"

                else:
                    if u_label in voiced_stops:
                        is_error = True
                        if comparison != f"{u_label}: ":
                            comparison += " | "
                        comparison += f"VOICING: No voicing detected - may be devoiced/voiceless"

            if not is_error:
                comparison += "No error"
            comparisons.append(comparison)

    return "\n".join(comparisons)
