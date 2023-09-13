//
// Created by Apple on 2023/9/12.
//

#ifndef SHOUXIECUDA_DOTMETHOD_H
#define SHOUXIECUDA_DOTMETHOD_H

#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <fstream>

class DotMethod
{
public:
    enum Enum{
        plus = 0,
        minus,
        equal,
        multiply,
        InvalidDotMethod = 11
    };

    DotMethod(void);
    DotMethod(Enum ee);
    explicit DotMethod(const std::string& ss);

    DotMethod& operator = (const DotMethod& cc);
    DotMethod& operator = (const std::string& ss);
    DotMethod& operator = (const Enum ee);

    bool operator == (const DotMethod& cc) const;
    bool operator == (const std::string& ss) const;
    bool operator == (const Enum ee) const;

    bool operator != (const DotMethod& cc) const;
    bool operator != (const std::string& ss) const;
    bool operator != (const Enum ee) const;

    inline std::string getString(void) const;
    inline Enum        getEnum(void) const;
    inline int         getValue(void) const;

    static Enum        fromString(std::string ss);
    static Enum        fromInt(int ss);
    static std::string toString(Enum ee);
    static int         toValue(Enum ee);

private:

    Enum        m_enum;
    std::string m_string;
    int         m_value;
};

#endif //SHOUXIECUDA_DOTMETHOD_H
