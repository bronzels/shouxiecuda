//
// Created by Apple on 2023/9/12.
//
#include <string>
#include <algorithm>

#include "dotmethod.h"

inline std::ostream& operator<< (std::ostream& _os, const DotMethod& _e)
{
    _os << _e.getString();
    return _os;
}

inline std::string DotMethod::getString(void) const
{
    return m_string;
}

DotMethod::Enum DotMethod::getEnum(void) const
{
    return m_enum;
}

int DotMethod::getValue(void) const
{
    return m_value;
}

DotMethod::DotMethod(void) : m_enum(plus), m_string("plus"), m_value(0)
{}

DotMethod::DotMethod(Enum _e) : m_enum(_e), m_string(toString(_e)), m_value(0)
{}

DotMethod::DotMethod(const std::string& _s) : m_enum(fromString(_s)), m_string(_s), m_value(toValue(m_enum))
{}

// Assignment operators
DotMethod& DotMethod::operator= (const DotMethod& _c)
{
    m_string = _c.m_string;
    m_enum = _c.m_enum;
    m_value = _c.m_value;
    return *this;
}

DotMethod& DotMethod::operator= (const std::string& _s)
{
    m_string = _s;
    m_enum = fromString(_s);
    m_value = toValue(m_enum);
    return *this;
}

DotMethod& DotMethod::operator= (Enum _e)
{
    m_enum = _e;
    m_string = toString(_e);
    m_value = toValue(_e);
    return *this;
}

bool DotMethod::operator== (const DotMethod& _c) const
{
    return (m_enum == _c.m_enum);
}

bool DotMethod::operator== (const std::string& _s) const
{
    return (m_string == _s);
}

bool DotMethod::operator== (const Enum _e) const
{
    return (m_enum == _e);
}

bool DotMethod::operator!= (const DotMethod& _c) const
{
    return (m_enum != _c.m_enum);
}

bool DotMethod::operator!= (const std::string& _s) const
{
    return (m_string != _s);
}

bool DotMethod::operator!= (const Enum _e) const
{
    return (m_enum != _e);
}

DotMethod::Enum DotMethod::fromString(std::string _s)
{
    // Case insensitive - make all upper case
    transform(_s.begin(), _s.end(), _s.begin(), toupper);
    if (_s == "")         return plus;
    else if (_s == "PLUS")    return plus;
    else if (_s == "MINUS")   return minus;
    else if (_s == "EQUAL") return equal;
    else if (_s == "MULTIPLY")  return multiply;
    throw std::range_error("Not a valid DotMethod value: " + _s);
    return InvalidDotMethod;
};

DotMethod::Enum DotMethod::fromInt(int _i)
{
    if (_i == toValue(plus))         return plus;
    else if (_i == toValue(minus))    return minus;
    else if (_i == toValue(equal))   return equal;
    else if (_i == toValue(multiply)) return multiply;
    throw std::range_error("Not a valid DotMethod value: " + std::to_string(_i));
    return InvalidDotMethod;
};

std::string DotMethod::toString(DotMethod::Enum _e)
{
    switch (_e) {
        case plus:    { return "plus";    }
        case minus:    { return "minus";    }
        case equal:   { return "equal";   }
        case multiply: { return "multiply"; }
    }
    return "InvalidDotMethod";
}

int DotMethod::toValue(DotMethod::Enum _e)
{
    switch (_e) {
        case plus:    { return 0; }
        case minus:    { return 1; }
        case equal:   { return 2; }
        case multiply: { return 3; }
    }
    return 11;  // Invalid
}
